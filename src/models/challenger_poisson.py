"""Challenger model: double Poisson regression for goals."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import statsmodels.api as sm
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "statsmodels is required. Install with `pip install statsmodels`."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class PoissonModelBundle:
    home_model: Any
    away_model: Any
    numeric_cols: list[str]
    categorical_cols: list[str]
    num_imputer: SimpleImputer
    num_scaler: StandardScaler | None
    cat_imputer: SimpleImputer | None
    encoder: OneHotEncoder | None
    max_goals: int = 10
    label_order: tuple[str, str, str] = ("W", "D", "L")

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        n_rows = len(df)
        X_num = np.empty((n_rows, 0))
        X_cat = np.empty((n_rows, 0))

        if self.numeric_cols:
            X_num = self.num_imputer.transform(df[self.numeric_cols])
            if self.num_scaler is not None:
                X_num = self.num_scaler.transform(X_num)

        if self.categorical_cols and self.encoder is not None and self.cat_imputer:
            X_cat_raw = self.cat_imputer.transform(df[self.categorical_cols])
            X_cat = self.encoder.transform(X_cat_raw)

        X_all = np.hstack([X_num, X_cat])
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        return sm.add_constant(X_all, has_constant="add")

    def predict_goal_rates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_const = self._design_matrix(df)
        lambda_home = self.home_model.predict(X_const)
        lambda_away = self.away_model.predict(X_const)
        return np.asarray(lambda_home), np.asarray(lambda_away)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        lambda_home, lambda_away = self.predict_goal_rates(df)
        probs = [
            outcome_probabilities(lh, la, max_goals=self.max_goals)
            for lh, la in zip(lambda_home, lambda_away)
        ]
        proba_df = pd.DataFrame(probs, columns=list(self.label_order))
        return proba_df


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def outcome_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for i in range(max_goals + 1):
        p_i = poisson_pmf(i, lambda_home)
        for j in range(max_goals + 1):
            p_j = poisson_pmf(j, lambda_away)
            p = p_i * p_j
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    total = p_home + p_draw + p_away
    if total == 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return p_home / total, p_draw / total, p_away / total


def fit_poisson_bundle(
    df: pd.DataFrame,
    feature_cols: list[str],
    home_goals_col: str = "home_score",
    away_goals_col: str = "away_score",
    max_goals: int = 10,
) -> PoissonModelBundle:
    X = df[feature_cols].copy()
    y_home = pd.to_numeric(df[home_goals_col], errors="coerce").fillna(0.0)
    y_away = pd.to_numeric(df[away_goals_col], errors="coerce").fillna(0.0)

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    filtered_numeric = []
    for col in numeric_cols:
        unique_count = X[col].nunique(dropna=True)
        if unique_count <= 1:
            logger.debug("Poisson: dropping constant column %s", col)
            continue
        filtered_numeric.append(col)
    numeric_cols = filtered_numeric
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    num_imputer = SimpleImputer(strategy="median")
    X_num = (
        num_imputer.fit_transform(X[numeric_cols])
        if numeric_cols
        else np.empty((len(X), 0))
    )
    num_scaler = None
    if numeric_cols:
        num_scaler = StandardScaler(with_mean=False)
        X_num = num_scaler.fit_transform(X_num)

    cat_imputer = None
    encoder = None
    X_cat = np.empty((len(X), 0))
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat_df = X[categorical_cols].astype(object)
        X_cat_raw = cat_imputer.fit_transform(X_cat_df)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = encoder.fit_transform(X_cat_raw)

    X_all = np.hstack([X_num, X_cat])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_const = sm.add_constant(X_all, has_constant="add")

    home_model = sm.GLM(y_home, X_const, family=sm.families.Poisson()).fit()
    away_model = sm.GLM(y_away, X_const, family=sm.families.Poisson()).fit()

    return PoissonModelBundle(
        home_model=home_model,
        away_model=away_model,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        num_imputer=num_imputer,
        num_scaler=num_scaler,
        cat_imputer=cat_imputer,
        encoder=encoder,
        max_goals=max_goals,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "elo_diff": rng.normal(0, 100, size=120),
            "fifa_rank_diff": rng.normal(0, 10, size=120),
            "home_score": rng.integers(0, 4, size=120),
            "away_score": rng.integers(0, 4, size=120),
            "confed": rng.choice(["UEFA", "CONMEBOL", "CAF"], size=120),
        }
    )
    bundle = fit_poisson_bundle(
        df,
        feature_cols=["elo_diff", "fifa_rank_diff", "confed"],
        home_goals_col="home_score",
        away_goals_col="away_score",
    )
    probs = bundle.predict_proba(df.head(5))
    logger.info("Poisson challenger ready. Sample probs:\n%s", probs)


if __name__ == "__main__":
    main()
