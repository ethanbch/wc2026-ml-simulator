"""Train champion and challenger models and save artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from src.models.challenger_lightgbm import build_lightgbm_classifier
from src.models.challenger_logistic import build_logistic_classifier
from src.models.challenger_poisson import fit_poisson_bundle
from src.models.champion_xgboost import build_xgb_classifier

from . import config

logger = logging.getLogger(__name__)

LABEL_CLASSES = ["W", "D", "L"]

DROP_COLUMNS = {
    "result",
    "date",
    "home_score",
    "away_score",
    "tournament",
    "match_id",
    "home_team",
    "away_team",
    "elo_home_post",
    "elo_away_post",
}

BANNED_PREFIXES = (
    "city",
    "country",
    "venue",
    "tournament",
)


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X = X.copy()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=list(DROP_COLUMNS), errors="ignore")
    X = X[[col for col in X.columns if not col.startswith(BANNED_PREFIXES)]]
    y = df["result"].astype(str)
    return X, y


def encode_labels(y: pd.Series) -> np.ndarray:
    label_map = {label: idx for idx, label in enumerate(LABEL_CLASSES)}
    y_enc = y.map(label_map)
    if y_enc.isna().any():
        unknown = y[y_enc.isna()].unique().tolist()
        raise ValueError(f"Unknown labels in target: {unknown}")
    return y_enc.to_numpy(dtype=int)


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    train_df = df[df[date_col] < pd.to_datetime(train_end)]
    val_df = df[
        (df[date_col] >= pd.to_datetime(train_end))
        & (df[date_col] < pd.to_datetime(val_end))
    ]
    test_df = df[df[date_col] >= pd.to_datetime(val_end)]
    return train_df, val_df, test_df


def train_models(
    dataset: Path = config.MATCH_DATASET,
    models_dir: Path = config.MODELS_DIR,
    train_end: str = "2022-01-01",
    val_end: str = "2024-01-01",
    use_val: bool = False,
    no_calibration: bool = False,
) -> None:
    df = pd.read_csv(dataset)
    train_df, val_df, _ = split_by_date(df, train_end, val_end)
    fit_df = pd.concat([train_df, val_df], ignore_index=True) if use_val else train_df

    fit_df = fit_df.sort_values("date").reset_index(drop=True)
    X_train, y_train = prepare_features(fit_df)
    y_train_enc = encode_labels(y_train)
    models_dir.mkdir(parents=True, exist_ok=True)

    champion_preprocess = build_preprocess(X_train)
    champion_pipeline = Pipeline(
        [
            ("preprocess", champion_preprocess),
            ("clf", build_xgb_classifier()),
        ]
    )

    if no_calibration:
        champion_model = champion_pipeline
    else:
        tscv = TimeSeriesSplit(n_splits=5)
        champion_model = CalibratedClassifierCV(
            estimator=champion_pipeline, method="isotonic", cv=tscv
        )
    champion_model.fit(X_train, y_train_enc)
    dump(champion_model, models_dir / "champion_xgboost.pkl")

    logistic_preprocess = build_preprocess(X_train)
    logistic_pipeline = Pipeline(
        [
            ("preprocess", logistic_preprocess),
            ("clf", build_logistic_classifier()),
        ]
    )
    logistic_pipeline.fit(X_train, y_train_enc)
    dump(logistic_pipeline, models_dir / "challenger_logistic.pkl")

    lightgbm_preprocess = build_preprocess(X_train)
    lightgbm_pipeline = Pipeline(
        [
            ("preprocess", lightgbm_preprocess),
            ("clf", build_lightgbm_classifier()),
        ]
    )
    lightgbm_pipeline.fit(X_train, y_train_enc)
    dump(lightgbm_pipeline, models_dir / "challenger_lightgbm.pkl")

    poisson_feature_candidates = [
        "elo_diff",
        "fifa_rank_diff",
        "fifa_points_diff",
        "match_importance",
        "neutral",
        "is_friendly",
        "days_rest_home",
        "days_rest_away",
        "home_form_goals_scored_8",
        "away_form_goals_scored_8",
        "home_form_goals_conceded_8",
        "away_form_goals_conceded_8",
        "home_form_win_rate_8",
        "away_form_win_rate_8",
        "home_form_clean_sheet_rate_8",
        "away_form_clean_sheet_rate_8",
        "home_fifa_rank",
        "away_fifa_rank",
        "home_fifa_points",
        "away_fifa_points",
        "home_confederation",
        "away_confederation",
        "home_squad_minutes_total",
        "away_squad_minutes_total",
        "home_squad_performance_gls",
        "away_squad_performance_gls",
        "home_squad_per_90_minutes_g_a",
        "away_squad_per_90_minutes_g_a",
    ]
    poisson_features = [
        col for col in poisson_feature_candidates if col in X_train.columns
    ]
    if not poisson_features:
        poisson_features = X_train.columns.tolist()
        logger.warning("Poisson feature fallback to full set")
    else:
        logger.info("Poisson features: %s", poisson_features)

    poisson_bundle = fit_poisson_bundle(
        fit_df,
        feature_cols=poisson_features,
        home_goals_col="home_score",
        away_goals_col="away_score",
    )
    poisson_bundle.label_order = tuple(LABEL_CLASSES)
    dump(poisson_bundle, models_dir / "challenger_poisson.pkl")

    feature_columns = X_train.columns.tolist()
    with open(models_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    with open(models_dir / "label_classes.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_CLASSES, f, indent=2)

    logger.info("Models saved to %s", models_dir)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--dataset", required=True, help="Match dataset CSV")
    parser.add_argument(
        "--models-dir", default=str(config.MODELS_DIR), help="Output dir"
    )
    parser.add_argument("--train-end", default="2022-01-01", help="Train end date")
    parser.add_argument("--val-end", default="2024-01-01", help="Validation end date")
    parser.add_argument(
        "--use-val",
        action="store_true",
        help="Include validation period in training",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable calibration for XGBoost champion",
    )
    args = parser.parse_args(argv)

    train_models(
        dataset=Path(args.dataset),
        models_dir=Path(args.models_dir),
        train_end=args.train_end,
        val_end=args.val_end,
        use_val=args.use_val,
        no_calibration=args.no_calibration,
    )


if __name__ == "__main__":
    main()
