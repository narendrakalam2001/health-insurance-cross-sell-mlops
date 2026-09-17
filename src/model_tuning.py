# ============================================================
# MODEL TUNING — Health Insurance Cross-Sell ML System
# ============================================================
# FIXED (see comments marked "FIX"): the original version searched
# hyperparameters with full 5-fold CV / 20 iterations directly on the
# full 243K-row training set, with SMOTENC re-computed on every single
# fold x iteration x model (up to ~1,400 SMOTENC calls total), AND had
# nested n_jobs=-1 in both the outer RandomizedSearchCV and several
# inner estimators (RandomForest, ExtraTrees, LightGBM, XGBoost,
# CatBoost) causing CPU oversubscription. Together these made a full
# run take 17+ hours. Both issues are fixed below.
# ============================================================

import numpy as np
import logging

from typing import Dict, List, Tuple

from sklearn.base              import BaseEstimator, clone
from sklearn.compose           import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection   import RandomizedSearchCV, StratifiedKFold, train_test_split

from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier, GradientBoostingClassifier,
                                   AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.naive_bayes   import GaussianNB, BernoulliNB
from xgboost               import XGBClassifier

from imblearn.pipeline     import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from src.config        import (RANDOM_STATE, N_JOBS, CV_FOLDS, RANDOM_SEARCH_ITERS,
                                SELECT_K, HYPERPARAM_SEARCH_SAMPLE_SIZE)
from src.preprocessing import safe_k

logger = logging.getLogger(__name__)


# ============================================================
# HELPER — compute safe n_iter for RandomizedSearchCV
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# MODEL GRIDS
# ============================================================

# Distance / linear models  →  need scaled input
scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LogisticRegression": (
        LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        {
            "classifier__penalty":      ["l1", "l2"],
            "classifier__C":            [0.01, 0.1, 1, 10],
            "classifier__solver":       ["liblinear", "saga"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "KNN": (
        # FIX: n_jobs pinned to 1 (see RandomForest comment below) —
        # KNN's own cost is now bounded by the search running on a
        # subsample (HYPERPARAM_SEARCH_SAMPLE_SIZE), not the full data.
        KNeighborsClassifier(n_jobs=1),
        {
            "classifier__n_neighbors": [3, 5, 9],
            "classifier__weights":     ["uniform", "distance"],
        }
    ),

    "SGD": (
        SGDClassifier(random_state=RANDOM_STATE, max_iter=2000, tol=1e-3),
        {
            "classifier__loss":         ["log_loss"],
            "classifier__alpha":        [1e-4, 1e-3, 1e-2],
            "classifier__penalty":      ["l2", "elasticnet"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "GaussianNB": (
        GaussianNB(), {}
    ),
}

# Tree-based models  →  work on raw / clipped input
# class_weight="balanced" + scale_pos_weight handle imbalance INSIDE model
# SMOTENC handles it at data level — both together = robust imbalance handling
unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__max_depth":        [5, 10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    ),

    "RandomForest": (
        # FIX: n_jobs=1 here (was N_JOBS=-1). The OUTER RandomizedSearchCV
        # already parallelizes across param combos x folds with n_jobs=-1.
        # Setting -1 on BOTH the outer search and this inner estimator
        # causes CPU oversubscription — every outer worker tries to spawn
        # its own full set of inner workers, and cores start thrashing
        # instead of doing useful work. This was a major contributor to
        # the 17+ hour runtime, especially on Windows (loky's process-
        # based backend has high spawn overhead, made much worse under
        # oversubscription).
        RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators":     [100, 200],
            "classifier__max_depth":        [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2],
        }
    ),

    "ExtraTrees": (
        # FIX: same n_jobs oversubscription fix as RandomForest above.
        ExtraTreesClassifier(n_jobs=1, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth":    [None, 10, 20],
        }
    ),

    "GradientBoosting": (
        # GradientBoosting has no class_weight → tuned via sample_weight / subsample
        # (also has no n_jobs param — inherently sequential boosting, so its
        # cost reduction comes entirely from the subsample-search fix below)
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [3, 5],
            "classifier__subsample":     [0.8, 1.0],
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":  [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 1.0],
        }
    ),

    "XGBoost": (
        # FIX: n_jobs=1 (was defaulting to all cores) — same oversubscription
        # fix as RandomForest/ExtraTrees above.
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
        {
            "classifier__n_estimators":    [100, 200],
            "classifier__learning_rate":   [0.05, 0.1],
            "classifier__max_depth":       [3, 5],
            "classifier__scale_pos_weight":[5, 7, 9],   # ~neg/pos ratio for ~12.3% positive rate
        }
    ),

    "BernoulliNB": (
        BernoulliNB(), {}
    ),
}

if LGBMClassifier is not None:
    unscaled_models["LightGBM"] = (
        # FIX: n_jobs=1 — same oversubscription fix.
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, is_unbalance=True, n_jobs=1),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [-1, 10],
            "classifier__num_leaves":    [31, 63],
        }
    )

if CatBoostClassifier is not None:
    unscaled_models["CatBoost"] = (
        # FIX: thread_count=1 — same oversubscription fix (CatBoost's
        # equivalent of n_jobs).
        CatBoostClassifier(verbose=0, random_state=RANDOM_STATE,
                            auto_class_weights="Balanced", thread_count=1),
        {
            "classifier__iterations":    [100, 200],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__depth":         [4, 6],
        }
    )


# ============================================================
# TUNE MODELS
# ============================================================

def tune_models(
    models:         Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor:   ColumnTransformer,
    cat_indices:    List[int],
    X_train:        "pd.DataFrame",
    y_train:        "pd.Series",
    use_smote:      bool = True,
    selector_k:     int  = SELECT_K
) -> Dict[str, ImbPipeline]:
    """
    For each model:
      preprocessor → [SMOTENC] → SelectKBest → classifier
    Tuned with RandomizedSearchCV (scoring = F1).

    FIX — large-dataset search strategy:
      If X_train has more than HYPERPARAM_SEARCH_SAMPLE_SIZE rows, the
      RandomizedSearchCV.fit() call runs on a stratified subsample of
      that size instead of the full data. This is what actually fixes
      the 17+ hour runtime: SMOTENC (the dominant per-fit cost on this
      dataset, due to high-cardinality categorical columns) was being
      recomputed on ~195K-row folds up to 100 times per model. On a
      40K-row subsample the same search is dramatically cheaper, and
      relative hyperparameter rankings barely change at this data
      volume. The FINAL model is then refit ONCE on the full X_train
      with the winning hyperparameters — so final model quality is
      unaffected; only the search cost drops.

    Returns dict of {model_name: best_pipeline} — pipelines are fully
    fit on the complete X_train/y_train, same contract as before.
    """

    final_pipelines: Dict[str, ImbPipeline] = {}

    k_safe = safe_k(selector_k, preprocessor, X_train)
    logger.info("Selector k set to %d (requested %d)", k_safe, selector_k)

    # ── FIX: subsample for the search phase on large datasets ────────
    n_total = len(X_train)
    if n_total > HYPERPARAM_SEARCH_SAMPLE_SIZE:
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train,
            train_size   = HYPERPARAM_SEARCH_SAMPLE_SIZE,
            stratify     = y_train,
            random_state = RANDOM_STATE
        )
        logger.info(
            "Large dataset (%d rows) — searching hyperparameters on a %d-row "
            "stratified subsample, then refitting best params on the full data",
            n_total, len(X_search)
        )
    else:
        X_search, y_search = X_train, y_train

    for name, (clf, param_dist) in models.items():

        logger.info("Tuning: %s", name)

        # ── Build pipeline steps ──────────────────────────────
        steps = [("preprocessor", preprocessor)]

        if use_smote and len(cat_indices) > 0:
            steps.append((
                "smote",
                SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)
            ))

        steps.append(("selector", SelectKBest(mutual_info_classif, k=k_safe)))
        steps.append(("classifier", clf))

        pipe = ImbPipeline(steps)

        # ── Randomized search (on subsample if large dataset) ─────
        if param_dist:
            n_iter = _compute_n_iter(param_dist, RANDOM_SEARCH_ITERS)

            search = RandomizedSearchCV(
                pipe,
                param_distributions = param_dist,
                n_iter              = n_iter,
                scoring             = "f1",
                cv                  = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                n_jobs              = N_JOBS,
                random_state        = RANDOM_STATE,
                verbose             = 0
            )
            search.fit(X_search, y_search)
            best_params = search.best_params_
            logger.info("%s best params (from search): %s", name, best_params)
        else:
            best_params = {}
            logger.info("%s has no hyperparameters to search", name)

        # ── FIX: refit winning params on the FULL training data ────
        # (previously the pipeline that came out of the search was used
        # directly — which was fine when search ran on full data, but
        # now that search runs on a subsample, we need one clean full-
        # data fit so the shipped model still sees all 243K rows)
        final_pipe = clone(pipe)
        if best_params:
            final_pipe.set_params(**best_params)
        final_pipe.fit(X_train, y_train)

        final_pipelines[name] = final_pipe

    return final_pipelines


# ============================================================
# NEURAL NETWORK — trained separately (no CV search)
# ============================================================

def train_mlp_pipeline(X_train, y_train, preprocessor, cat_indices: List[int]):
    """
    MLP trained separately outside RandomizedSearchCV.
    Reason: MLP training time makes CV search impractical.
    Already trains once on full data — no change needed here for the
    large-dataset fix (it was never part of the 17-hour bottleneck).
    """
    from sklearn.neural_network import MLPClassifier

    logger.info("Training Neural Network (MLP) ...")

    pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",         SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)),
        ("classifier",    MLPClassifier(
            hidden_layer_sizes   = (128, 64),
            activation           = "relu",
            solver               = "adam",
            alpha                = 0.0001,
            batch_size           = 512,
            learning_rate        = "adaptive",
            max_iter             = 50,
            early_stopping       = True,
            validation_fraction  = 0.1,
            n_iter_no_change     = 5,
            random_state         = RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)

    logger.info("MLP training done")

    return pipe