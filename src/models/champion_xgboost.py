"""Champion model: calibrated XGBoost classifier."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "xgboost is required. Install with `pip install xgboost`."
    ) from exc

logger = logging.getLogger(__name__)


def build_xgb_classifier(
    n_estimators: int = 500,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
    )


def fit_champion(
    X,
    y,
    calibrate: bool = True,
    method: str = "isotonic",
    cv: int = 5,
    **kwargs,
):
    base_model = build_xgb_classifier(**kwargs)
    if calibrate:
        model = CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)
    else:
        model = base_model
    model.fit(X, y)
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 8))
    y = rng.choice(["W", "D", "L"], size=120)
    model = fit_champion(X, y, calibrate=False)
    logger.info("Champion model trained: %s", type(model).__name__)


if __name__ == "__main__":
    main()
