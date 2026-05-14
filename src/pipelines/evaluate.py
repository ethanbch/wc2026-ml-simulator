"""Evaluate trained models on the test split with robust label alignment."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import compute_classification_metrics
from src.models.challenger_poisson import PoissonModelBundle

from . import config

logger = logging.getLogger(__name__)

DROP_COLUMNS = {
    "result",
    "date",
    "home_score",
    "away_score",
    "tournament",
    "match_id",
    "home_team",
    "away_team",
}


def split_by_date(
    df: pd.DataFrame, train_end: str, val_end: str, date_col: str = "date"
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


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df.drop(columns=list(DROP_COLUMNS), errors="ignore")
    X_final = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in X.columns:
            X_final[col] = X[col]
        else:
            X_final[col] = 0

    for col in X_final.select_dtypes(include=["bool"]).columns:
        X_final[col] = X_final[col].astype(int)
    return X_final


def load_json(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_models(
    dataset: Path = config.MATCH_DATASET,
    models_dir: Path = config.MODELS_DIR,
    train_end: str = "2022-01-01",
    val_end: str = "2024-01-01",
    output: Path = config.PROC_DIR / "metrics.csv",
) -> pd.DataFrame:
    df = pd.read_csv(dataset)
    _, _, test_df = split_by_date(df, train_end, val_end)

    feature_columns = load_json(models_dir / "feature_columns.json")
    label_classes = load_json(models_dir / "label_classes.json")

    ordered_labels = sorted(label_classes)

    X_test = prepare_features(test_df, feature_columns)
    y_test = test_df["result"].astype(str).tolist()

    results = []
    model_paths = {
        "champion_xgboost": models_dir / "champion_xgboost.pkl",
        "challenger_logistic": models_dir / "challenger_logistic.pkl",
        "challenger_lightgbm": models_dir / "challenger_lightgbm.pkl",
        "challenger_poisson": models_dir / "challenger_poisson.pkl",
    }

    plots_dir = Path("data/processed/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    for name, path in model_paths.items():
        if not path.exists():
            logger.warning("Model not found: %s", path)
            continue

        model = load(path)

        if isinstance(model, PoissonModelBundle):
            proba_df = model.predict_proba(X_test)
            y_proba = proba_df[ordered_labels].to_numpy()
        else:
            y_proba_raw = model.predict_proba(X_test)

            model_classes = list(model.classes_)

            y_proba_dict = {}
            for i, cls in enumerate(model_classes):
                label = label_classes[cls] if isinstance(cls, (int, np.integer)) else cls
                y_proba_dict[label] = y_proba_raw[:, i]

            y_proba = np.column_stack([y_proba_dict[l] for l in ordered_labels])

        metrics = compute_classification_metrics(y_test, y_proba, ordered_labels)
        metrics["model"] = name
        results.append(metrics)

        y_pred = [ordered_labels[i] for i in np.argmax(y_proba, axis=1)]
        viz_order = ["W", "D", "L"]
        cm = confusion_matrix(y_test, y_pred, labels=viz_order)

        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=viz_order,
            yticklabels=viz_order,
        )
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(plots_dir / f"cm_{name}.png", bbox_inches="tight")
        plt.close()

    results_df = pd.DataFrame(results).sort_values("log_loss")
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)
    logger.info("Evaluation finished with label realignment.")
    print(results_df[["model", "accuracy", "log_loss"]])
    return results_df


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--dataset", required=True, help="Match dataset CSV")
    parser.add_argument(
        "--models-dir", default=str(config.MODELS_DIR), help="Models dir"
    )
    parser.add_argument("--train-end", default="2022-01-01", help="Train end date")
    parser.add_argument("--val-end", default="2024-01-01", help="Validation end date")
    parser.add_argument(
        "--output", default=str(config.PROC_DIR / "metrics.csv"), help="Output metrics CSV"
    )
    args = parser.parse_args(argv)

    evaluate_models(
        dataset=Path(args.dataset),
        models_dir=Path(args.models_dir),
        train_end=args.train_end,
        val_end=args.val_end,
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
