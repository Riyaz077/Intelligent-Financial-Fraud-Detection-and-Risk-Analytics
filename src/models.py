"""Candidate models for imbalanced binary fraud classification."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from .preprocessing import build_preprocessor

def candidate_estimators(random_state=42):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=random_state, n_jobs=-1
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=random_state
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state,
            n_jobs=-1
        )
    except ImportError:
        pass
    return models

def build_pipelines(X, random_state=42):
    prep = build_preprocessor(X)
    return {
        name: Pipeline([("preprocess", prep), ("classifier", estimator)])
        for name, estimator in candidate_estimators(random_state).items()
    }
