"""Leakage-aware model training and imbalanced-class evaluation."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from .models import build_pipelines

def evaluate_models(X, y, test_size=0.2, random_state=42, threshold=0.5):
    counts = y.value_counts()
    if counts.min() < 2:
        raise ValueError("Each class needs at least two records for a stratified split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Treat the minority class as the positive class for this binary fraud task.
    positive_label = counts.idxmin()
    y_test_binary = (y_test == positive_label).astype(int)
    results, fitted = [], {}
    for name, pipeline in build_pipelines(X_train, random_state).items():
        pipeline.fit(X_train, y_train)
        if not hasattr(pipeline, "predict_proba"):
            continue
        classes = list(pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index(positive_label)
        probabilities = pipeline.predict_proba(X_test)[:, positive_index]
        pred_binary = (probabilities >= threshold).astype(int)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test_binary, pred_binary),
            "Precision": precision_score(y_test_binary, pred_binary, zero_division=0),
            "Recall": recall_score(y_test_binary, pred_binary, zero_division=0),
            "F1": f1_score(y_test_binary, pred_binary, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test_binary, probabilities),
            "PR_AUC": average_precision_score(y_test_binary, probabilities),
        })
        fitted[name] = {
            "pipeline": pipeline, "X_test": X_test, "y_test_binary": y_test_binary,
            "probabilities": probabilities, "positive_label": positive_label
        }
    if not results:
        raise RuntimeError("No probability-capable classifier was available.")
    metrics = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
    return metrics, fitted

def threshold_metrics(y_true, probabilities, threshold=0.5):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp)
    }
