"""Bookmaker benchmark helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.clip(odds, 1e-6, None)
    return inv / inv.sum(axis=1, keepdims=True)


def bookmaker_log_loss(y_true: list[str], odds: np.ndarray, labels: list[str]) -> float:
    probs = implied_probabilities(odds)
    return float(log_loss(y_true, probs, labels=labels))
