"""Build match-level features for model training."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .elo_calculator import add_elo_features
from .rolling_form import add_rolling_form_features

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_match_result_labels(
    df: pd.DataFrame,
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
) -> pd.DataFrame:
    df = df.copy()

    def _label(row: pd.Series) -> str:
        if row[home_score_col] > row[away_score_col]:
            return "W"
        if row[home_score_col] < row[away_score_col]:
            return "L"
        return "D"

    df["result"] = df.apply(_label, axis=1)
    return df


def tournament_importance(tournament: str | None) -> float:
    if not tournament:
        return 1.0
    t = str(tournament).lower()
    if "world cup" in t:
        return 1.5
    if "qual" in t:
        return 1.1
    if "friendly" in t or "exhibition" in t:
        return 0.3
    return 1.0


def add_match_importance(
    df: pd.DataFrame, tournament_col: str = "tournament"
) -> pd.DataFrame:
    df = df.copy()
    df["match_importance"] = df[tournament_col].apply(tournament_importance)
    return df


def add_days_rest(
    df: pd.DataFrame,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_rest_order"] = range(len(df))
    df_sorted = df.sort_values([date_col, "_rest_order"]).copy()

    last_played: dict[str, pd.Timestamp] = {}
    home_rest = []
    away_rest = []

    for _, row in df_sorted.iterrows():
        home_team = row[home_col]
        away_team = row[away_col]
        match_date = row[date_col]

        home_rest.append(
            (match_date - last_played[home_team]).days
            if home_team in last_played
            else None
        )
        away_rest.append(
            (match_date - last_played[away_team]).days
            if away_team in last_played
            else None
        )

        last_played[home_team] = match_date
        last_played[away_team] = match_date

    df_sorted["days_rest_home"] = home_rest
    df_sorted["days_rest_away"] = away_rest
    df_sorted = df_sorted.drop(columns=["_rest_order"]).sort_index()
    return df_sorted


def _merge_rankings_asof(
    df: pd.DataFrame,
    rankings: pd.DataFrame,
    team_col: str,
    prefix: str,
) -> pd.DataFrame:
    left = df[[team_col, "date"]].copy()
    left["row_id"] = df.index
    left = left.rename(columns={team_col: "team"}).sort_values("date")

    rankings = rankings.copy()
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"], errors="coerce")
    rankings = rankings.sort_values("rank_date")

    merged = pd.merge_asof(
        left,
        rankings,
        left_on="date",
        right_on="rank_date",
        by="team",
        direction="backward",
    )
    for col in ["rank", "points", "confederation"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged.set_index("row_id").reindex(df.index)

    rename_cols = {
        "rank": f"{prefix}fifa_rank",
        "points": f"{prefix}fifa_points",
        "confederation": f"{prefix}confederation",
    }
    return merged.rename(columns=rename_cols)[list(rename_cols.values())]


def add_fifa_rank_features(
    df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    df = df.copy()
    home_rank = _merge_rankings_asof(df, rankings_df, home_col, prefix="home_")
    away_rank = _merge_rankings_asof(df, rankings_df, away_col, prefix="away_")

    df = df.join(home_rank).join(away_rank)
    df["fifa_rank_diff"] = df["home_fifa_rank"] - df["away_fifa_rank"]
    df["fifa_points_diff"] = df["home_fifa_points"] - df["away_fifa_points"]
    df["same_confederation"] = df["home_confederation"] == df["away_confederation"]
    return df


def merge_team_features(
    df: pd.DataFrame,
    team_features: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    df = df.copy()
    team_features = team_features.copy()
    if "team" not in team_features.columns:
        raise ValueError("Team features must include a 'team' column")

    home_features = team_features.add_prefix("home_").rename(
        columns={"home_team": "team"}
    )
    df = df.merge(home_features, left_on=home_col, right_on="team", how="left")
    df = df.drop(columns=["team"])

    away_features = team_features.add_prefix("away_").rename(
        columns={"away_team": "team"}
    )
    df = df.merge(away_features, left_on=away_col, right_on="team", how="left")
    df = df.drop(columns=["team"])
    return df


def build_match_dataset(
    matches_df: pd.DataFrame,
    fifa_rankings_df: pd.DataFrame | None = None,
    team_features_df: pd.DataFrame | None = None,
    form_n: int = 8,
    form_decay: float = 0.85,
) -> pd.DataFrame:
    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = add_match_result_labels(df)
    df = add_match_importance(df)
    df = add_elo_features(df)
    df = add_rolling_form_features(df, n=form_n, decay=form_decay)
    df = add_days_rest(df)

    if fifa_rankings_df is not None:
        df = add_fifa_rank_features(df, fifa_rankings_df)

    if team_features_df is not None:
        df = merge_team_features(df, team_features_df)

    return df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build match-level dataset")
    parser.add_argument("--matches-csv", default=None, help="Path to match results CSV")
    parser.add_argument(
        "--fifa-rankings", default=None, help="Path to FIFA rankings CSV"
    )
    parser.add_argument(
        "--team-features", default=None, help="Path to team features CSV"
    )
    parser.add_argument("--out", default=None, help="Output dataset CSV path")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if not args.matches_csv:
        parser.print_help()
        return

    matches_df = pd.read_csv(Path(args.matches_csv))
    fifa_rankings_df = (
        pd.read_csv(Path(args.fifa_rankings)) if args.fifa_rankings else None
    )
    team_features_df = (
        pd.read_csv(Path(args.team_features)) if args.team_features else None
    )

    dataset = build_match_dataset(matches_df, fifa_rankings_df, team_features_df)

    output_path = (
        Path(args.out)
        if args.out
        else get_project_root() / "data" / "processed" / "match_dataset.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.info("Saved match dataset to %s", output_path)


if __name__ == "__main__":
    main()
