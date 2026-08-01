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


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
    """Compute the full suite of evaluation metrics."""
    return {
        "pr_auc": pr_auc(y_true, y_scores),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "f2": f2(y_true, y_pred),
        "mcc": mcc(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["bad", "good"], zero_division=0
        ),
    }
