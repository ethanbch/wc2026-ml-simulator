"""Rolling form features with exponential decay."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _weighted_stats(history: list[dict], n: int, decay: float) -> dict[str, float]:
    if not history:
        return {
            "goals_scored": np.nan,
            "goals_conceded": np.nan,
            "win_rate": np.nan,
            "clean_sheet_rate": np.nan,
        }

    recent = history[-n:]
    weights = np.array([decay**i for i in range(len(recent) - 1, -1, -1)], dtype=float)
    weights = weights / weights.sum()

    goals_scored = np.array([row["goals_scored"] for row in recent], dtype=float)
    goals_conceded = np.array([row["goals_conceded"] for row in recent], dtype=float)
    wins = np.array([row["win"] for row in recent], dtype=float)
    clean_sheets = np.array([row["clean_sheet"] for row in recent], dtype=float)

    return {
        "goals_scored": float(np.sum(goals_scored * weights)),
        "goals_conceded": float(np.sum(goals_conceded * weights)),
        "win_rate": float(np.sum(wins * weights)),
        "clean_sheet_rate": float(np.sum(clean_sheets * weights)),
    }


def add_rolling_form_features(
    matches_df: pd.DataFrame,
    n: int = 8,
    decay: float = 0.85,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
) -> pd.DataFrame:
    df = matches_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_form_order"] = range(len(df))
    df_sorted = df.sort_values([date_col, "_form_order"]).copy()

    history: dict[str, list[dict]] = defaultdict(list)
    home_features = []
    away_features = []

    for _, row in df_sorted.iterrows():
        home_team = row[home_col]
        away_team = row[away_col]

        home_stats = _weighted_stats(history[home_team], n=n, decay=decay)
        away_stats = _weighted_stats(history[away_team], n=n, decay=decay)

        home_features.append(home_stats)
        away_features.append(away_stats)

        home_score = row[home_score_col]
        away_score = row[away_score_col]

        home_win = 1.0 if home_score > away_score else 0.0
        away_win = 1.0 if away_score > home_score else 0.0

        history[home_team].append(
            {
                "goals_scored": home_score,
                "goals_conceded": away_score,
                "win": home_win,
                "clean_sheet": 1.0 if away_score == 0 else 0.0,
            }
        )
        history[away_team].append(
            {
                "goals_scored": away_score,
                "goals_conceded": home_score,
                "win": away_win,
                "clean_sheet": 1.0 if home_score == 0 else 0.0,
            }
        )

    home_df = pd.DataFrame(home_features, index=df_sorted.index)
    away_df = pd.DataFrame(away_features, index=df_sorted.index)

    for col in home_df.columns:
        df_sorted[f"home_form_{col}_{n}"] = home_df[col]
        df_sorted[f"away_form_{col}_{n}"] = away_df[col]

    df_sorted = df_sorted.drop(columns=["_form_order"])
    df_sorted = df_sorted.sort_index()
    return df_sorted


def compute_latest_form_features(
    matches_df: pd.DataFrame,
    n: int = 8,
    decay: float = 0.85,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
) -> pd.DataFrame:
    df = matches_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_form_order"] = range(len(df))
    df_sorted = df.sort_values([date_col, "_form_order"]).copy()

    history: dict[str, list[dict]] = defaultdict(list)

    for _, row in df_sorted.iterrows():
        home_team = row[home_col]
        away_team = row[away_col]
        home_score = row[home_score_col]
        away_score = row[away_score_col]

        history[home_team].append(
            {
                "goals_scored": home_score,
                "goals_conceded": away_score,
                "win": 1.0 if home_score > away_score else 0.0,
                "clean_sheet": 1.0 if away_score == 0 else 0.0,
            }
        )
        history[away_team].append(
            {
                "goals_scored": away_score,
                "goals_conceded": home_score,
                "win": 1.0 if away_score > home_score else 0.0,
                "clean_sheet": 1.0 if home_score == 0 else 0.0,
            }
        )

    rows = []
    for team, hist in history.items():
        stats = _weighted_stats(hist, n=n, decay=decay)
        rows.append(
            {
                "team": team,
                f"form_goals_scored_{n}": stats["goals_scored"],
                f"form_goals_conceded_{n}": stats["goals_conceded"],
                f"form_win_rate_{n}": stats["win_rate"],
                f"form_clean_sheet_rate_{n}": stats["clean_sheet_rate"],
            }
        )

    return pd.DataFrame(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute rolling form features")
    parser.add_argument("--matches-csv", default=None, help="Path to match results CSV")
    parser.add_argument("--out", default=None, help="Output CSV with form features")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if not args.matches_csv:
        parser.print_help()
        return

    matches_path = Path(args.matches_csv)
    df = pd.read_csv(matches_path)
    df = add_rolling_form_features(df)

    output_path = (
        Path(args.out)
        if args.out
        else get_project_root() / "data" / "processed" / "matches_with_form.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved match form features to %s", output_path)


if __name__ == "__main__":
    main()
