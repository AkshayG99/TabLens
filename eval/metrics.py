import numpy as np
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Precision-Recall AUC — primary metric for imbalanced credit risk data."""
    return average_precision_score(y_true, y_scores)


def f2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F2 Score — emphasizes recall over precision (critical for catching defaults)."""
    return fbeta_score(y_true, y_pred, beta=2)


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient — balanced measure even with class imbalance."""
    return matthews_corrcoef(y_true, y_pred)


def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted F1 (per-class F1 averaged by class support) — TabReason/FinBen's metric."""
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
    """Compute the full suite of evaluation metrics.

    Guards against a small/unlucky --limit sample containing only one class:
    roc_auc_score raises outright with a single class in y_true (undefined --
    there's no negative class to rank against), and without an explicit
    `labels=[0, 1]`, confusion_matrix/classification_report both infer their
    label set from whatever classes are actually present, silently
    collapsing to a 1x1 matrix or a mismatched report instead of the 2x2
    shape every caller (e.g. the cm[0][0]/cm[0][1]/... indexing in
    eval/evaluate.py) assumes.
    """
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y_true, y_scores)
    return {
        "pr_auc": pr_auc(y_true, y_scores),
        "roc_auc": roc_auc,
        "f2": f2(y_true, y_pred),
        "mcc": mcc(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_weighted": f1_weighted(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["bad", "good"], zero_division=0
        ),
    }
