"""Compute Elo ratings and match-level features."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def expected_score(
    elo_home: float, elo_away: float, home_advantage: float = 100.0
) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_away - (elo_home + home_advantage)) / 400.0))


def goal_diff_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    if goal_diff == 3:
        return 1.75
    if goal_diff == 4:
        return 1.9
    return 2.0


def k_factor(tournament: str | None, default_k: float = 20.0) -> float:
    if not tournament:
        return default_k
    t = str(tournament).lower()
    if "world cup" in t and "qual" not in t:
        return 40.0
    if "qual" in t:
        return 30.0
    if "nations league" in t:
        return 20.0
    if "cup" in t or "championship" in t:
        return 20.0
    if "friendly" in t or "exhibition" in t:
        return 10.0
    return default_k


def result_from_scores(home_score: float, away_score: float) -> float:
    if pd.isna(home_score) or pd.isna(away_score):
        return 0.5
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def add_elo_features(
    matches_df: pd.DataFrame,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    tournament_col: str = "tournament",
    neutral_col: str = "neutral",
    base_elo: float = 1500.0,
    home_advantage: float = 100.0,
    return_history: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    df = matches_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_elo_order"] = range(len(df))
    df_sorted = df.sort_values([date_col, "_elo_order"]).copy()

    ratings: dict[str, float] = defaultdict(lambda: base_elo)
    records = []
    history = []

    for idx, row in df_sorted.iterrows():
        home_team = row[home_col]
        away_team = row[away_col]
        elo_home = ratings[home_team]
        elo_away = ratings[away_team]

        is_neutral = False
        if neutral_col in row.index:
            is_neutral = bool(row[neutral_col])
        adv = 0.0 if is_neutral else home_advantage

        expected_home = expected_score(elo_home, elo_away, adv)
        home_score = row[home_score_col]
        away_score = row[away_score_col]
        actual_home = result_from_scores(home_score, away_score)
        if pd.isna(home_score) or pd.isna(away_score):
            goal_diff = 0
        else:
            goal_diff = int(abs(home_score - away_score))

        k = k_factor(row.get(tournament_col))
        k_adj = k * goal_diff_multiplier(goal_diff)

        new_elo_home = elo_home + k_adj * (actual_home - expected_home)
        new_elo_away = elo_away + k_adj * ((1.0 - actual_home) - (1.0 - expected_home))

        ratings[home_team] = new_elo_home
        ratings[away_team] = new_elo_away

        records.append(
            {
                "row_id": idx,
                "elo_home": elo_home,
                "elo_away": elo_away,
                "elo_diff": elo_home - elo_away,
                "elo_home_post": new_elo_home,
                "elo_away_post": new_elo_away,
            }
        )

        if return_history:
            history.append(
                {
                    "date": row[date_col],
                    "team": home_team,
                    "opponent": away_team,
                    "elo_before": elo_home,
                    "elo_after": new_elo_home,
                    "tournament": row.get(tournament_col),
                }
            )
            history.append(
                {
                    "date": row[date_col],
                    "team": away_team,
                    "opponent": home_team,
                    "elo_before": elo_away,
                    "elo_after": new_elo_away,
                    "tournament": row.get(tournament_col),
                }
            )

    records_df = pd.DataFrame(records).set_index("row_id")
    df = df.join(records_df, how="left")
    df = df.drop(columns=["_elo_order"])

    if return_history:
        history_df = pd.DataFrame(history)
        return df, history_df
    return df


def compute_latest_elo(
    matches_df: pd.DataFrame,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    tournament_col: str = "tournament",
    neutral_col: str = "neutral",
    base_elo: float = 1500.0,
    home_advantage: float = 100.0,
) -> pd.DataFrame:
    _, history_df = add_elo_features(
        matches_df,
        date_col=date_col,
        home_col=home_col,
        away_col=away_col,
        home_score_col=home_score_col,
        away_score_col=away_score_col,
        tournament_col=tournament_col,
        neutral_col=neutral_col,
        base_elo=base_elo,
        home_advantage=home_advantage,
        return_history=True,
    )
    history_df = history_df.sort_values("date")
    latest = history_df.groupby("team").tail(1)
    latest = latest[["team", "elo_after"]].rename(columns={"elo_after": "elo"})
    return latest.reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Elo ratings")
    parser.add_argument("--matches-csv", default=None, help="Path to match results CSV")
    parser.add_argument("--out", default=None, help="Output CSV path for Elo history")
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
    df, history = add_elo_features(df, return_history=True)

    output_path = (
        Path(args.out)
        if args.out
        else get_project_root() / "data" / "processed" / "elo_ratings.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_path, index=False)
    logger.info("Saved Elo history to %s", output_path)


if __name__ == "__main__":
    main()
