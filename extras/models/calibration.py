"""Calibration helpers for probabilistic classifiers."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def calibrate_classifier(
    estimator,
    X,
    y,
    method: str = "isotonic",
    cv: int = 5,
) -> CalibratedClassifierCV:
    model = CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    model.fit(X, y)
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 6))
    y = rng.integers(0, 3, size=120)
    base = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model = calibrate_classifier(base, X, y, method="sigmoid", cv=3)
    logger.info("Calibrated model ready: %s", type(model).__name__)


if __name__ == "__main__":
    main()
