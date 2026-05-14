"""Challenger model: LightGBM classifier."""

from __future__ import annotations

import logging

import numpy as np

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "lightgbm is required. Install with `pip install lightgbm`."
    ) from exc

logger = logging.getLogger(__name__)


def build_lightgbm_classifier(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    random_state: int = 42,
) -> "lgb.LGBMClassifier":
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(21)
    X = rng.normal(size=(90, 6))
    y = rng.choice(["W", "D", "L"], size=90)
    model = build_lightgbm_classifier()
    model.fit(X, y)
    logger.info("LightGBM challenger trained: %s", type(model).__name__)


if __name__ == "__main__":
    main()
