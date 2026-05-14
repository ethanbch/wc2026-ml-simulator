"""SHAP analysis helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

try:
    import shap
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("shap is required. Install with `pip install shap`.") from exc

logger = logging.getLogger(__name__)


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 1000):
    sample = X.sample(n=min(len(X), max_samples), random_state=42)
    explainer = shap.Explainer(model, sample)
    shap_values = explainer(sample)
    return shap_values, sample


def shap_summary_plot(
    model,
    X: pd.DataFrame,
    output_path: Path,
    max_samples: int = 1000,
) -> None:
    import matplotlib.pyplot as plt

    shap_values, sample = compute_shap_values(model, X, max_samples=max_samples)
    shap.summary_plot(shap_values, sample, show=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved SHAP summary plot to %s", output_path)
