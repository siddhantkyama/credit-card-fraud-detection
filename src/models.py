"""The models we compare. Each one is a scikit-learn compatible estimator/pipeline.

The comparison answers the interview question "why did you pick this model?"
with evidence instead of habit.
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42

# Models that must NOT be deployed (used for comparison only).
COMPARISON_ONLY = {"Isolation Forest (unsupervised)"}


class IsolationForestScorer(ClassifierMixin, BaseEstimator):
    """Unsupervised anomaly detector wrapped so it can be scored like a classifier.

    It never sees the fraud labels. It just learns what "normal" looks like and
    scores how unusual each transaction is. Useful as a sanity check: how much
    do the labels actually help?
    """

    def __init__(self, n_estimators=300, random_state=SEED):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
        ).fit(X)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        # score_samples is higher for "more normal"; flip it so higher = more suspicious.
        return -self.model_.score_samples(X)


def make_xgb(pos_weight: float = 1.0, **overrides) -> XGBClassifier:
    params = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        tree_method="hist", eval_metric="aucpr",
        n_jobs=-1, random_state=SEED, verbosity=0,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def make_lgbm(pos_weight: float = 1.0, **overrides) -> LGBMClassifier:
    params = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        n_jobs=-1, random_state=SEED, verbose=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


# Boosting models that can be tuned with Optuna.
BOOSTERS = {"XGBoost": make_xgb, "LightGBM": make_lgbm}


def build_models(pos_weight: float) -> dict:
    """Return every candidate model.

    pos_weight ~ sqrt(#normal / #fraud). The full ratio (~577) over-weights the
    few fraud rows and hurts precision; the square root is a common compromise.
    """
    return {
        # 1. A simple, fast, explainable baseline. Every project needs one.
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        # 2. Your original model, but handling imbalance with class weights.
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED,
        ),
        # 3. Your original approach (Random Forest + SMOTE). SMOTE sits INSIDE the
        #    pipeline, so during cross-validation it only touches training folds.
        #    sampling_strategy=0.1 -> fraud becomes 10% of normal (full 1:1 is slow).
        "Random Forest + SMOTE": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.1, random_state=SEED)),
            ("clf", RandomForestClassifier(
                n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
        ]),
        # 4-6. Gradient boosting: the usual winner on tabular fraud data.
        "XGBoost": make_xgb(pos_weight),
        "LightGBM": make_lgbm(pos_weight),
        "LightGBM + SMOTE": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.1, random_state=SEED)),
            ("clf", make_lgbm(pos_weight=1.0)),
        ]),
        # 7. Unsupervised anomaly detection (no labels used).
        "Isolation Forest (unsupervised)": IsolationForestScorer(),
    }
