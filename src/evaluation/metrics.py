"""Metrics for multi-class W/D/L classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def _one_hot(y_true: list[str], labels: list[str]) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y_idx = np.array([label_to_index[label] for label in y_true], dtype=int)
    y_one_hot = np.zeros((len(y_true), len(labels)), dtype=float)
    y_one_hot[np.arange(len(y_true)), y_idx] = 1.0
    return y_one_hot


def multiclass_brier_score(
    y_true: list[str],
    y_proba: np.ndarray,
    labels: list[str],
) -> float:
    y_one_hot = _one_hot(y_true, labels)
    return float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))


def compute_classification_metrics(
    y_true: list[str],
    y_proba: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    y_pred = [labels[int(idx)] for idx in np.argmax(y_proba, axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
        "brier_score": multiclass_brier_score(y_true, y_proba, labels),
    }
