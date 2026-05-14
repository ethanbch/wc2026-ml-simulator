"""Challenger model: multinomial logistic regression."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def build_logistic_classifier(
    max_iter: int = 5000,
    c_value: float = 1.0,
) -> LogisticRegression:
    return LogisticRegression(
        max_iter=max_iter,
        C=c_value,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 5))
    y = rng.choice(["W", "D", "L"], size=80)
    model = build_logistic_classifier()
    model.fit(X, y)
    logger.info("Logistic challenger trained: %s", type(model).__name__)


if __name__ == "__main__":
    main()
