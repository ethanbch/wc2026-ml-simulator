"""Aggregate club player stats into national team features."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE = {"player", "team", "nation", "squad", "comp", "pos", "position"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_player_name(name: str) -> str:
    if pd.isna(name):
        return ""
    return " ".join(str(name).strip().split())


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(part) for part in col if part]) for col in df.columns
        ]
    return df


def normalize_feature_name(name: str) -> str:
    name = name.strip().lower().replace("%", "pct")
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def prepare_player_stats(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(player_stats_df).copy()
    rename_map = {}
    for col in df.columns:
        if col.lower() == "player":
            rename_map[col] = "player"
    df = df.rename(columns=rename_map)

    if "player" not in df.columns:
        raise ValueError("Player column not found in player stats")

    df["player"] = df["player"].apply(normalize_player_name)
    return df


def aggregate_national_team_stats(
    squad_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    weight_col: str = "90s",
    metrics: Iterable[str] | None = None,
    min_minutes: float = 0.0,
) -> pd.DataFrame:
    if "team" not in squad_df.columns or "player" not in squad_df.columns:
        raise ValueError("Squad data must include 'team' and 'player' columns")

    squad_df = squad_df.copy()
    squad_df["player"] = squad_df["player"].apply(normalize_player_name)
    squad_df["team"] = squad_df["team"].apply(normalize_player_name)

    player_stats_df = prepare_player_stats(player_stats_df)
    merged = squad_df.merge(
        player_stats_df.drop(columns=["team"], errors="ignore"), on="player", how="left"
    )

    if weight_col not in merged.columns:
        if "minutes" in merged.columns:
            weight_col = "minutes"
        else:
            weight_col = None

    results = []
    for team, group in merged.groupby("team"):
        weights = None
        if weight_col:
            weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0)
            if min_minutes > 0:
                group = group[weights >= min_minutes]
                weights = weights[weights >= min_minutes]
        if group.empty:
            continue

        minutes_total = (
            float(pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0).sum())
            if weight_col
            else float(len(group))
        )

        if weights is None or weights.sum() == 0:
            weights = pd.Series(np.ones(len(group)), index=group.index)
        weights = weights / weights.sum()

        numeric_cols = group.select_dtypes(include=["number"]).columns.tolist()
        if metrics is not None:
            metrics_set = set(metrics)
            numeric_cols = [col for col in numeric_cols if col in metrics_set]

        if weight_col and weight_col in numeric_cols:
            numeric_cols.remove(weight_col)

        agg_row = {"team": team}
        agg_row["squad_minutes_total"] = minutes_total

        for col in numeric_cols:
            if col in DEFAULT_EXCLUDE:
                continue
            values = pd.to_numeric(group[col], errors="coerce").fillna(0.0)
            feature_name = normalize_feature_name(col)
            if feature_name == "age":
                feature_name = "avg_age"
            agg_row[f"squad_{feature_name}"] = float(np.sum(values * weights))

        results.append(agg_row)

    return pd.DataFrame(results)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate FBref player stats to national teams"
    )
    parser.add_argument("--squad-csv", default=None, help="Path to squad CSV")
    parser.add_argument("--player-stats", default=None, help="Path to player stats CSV")
    parser.add_argument("--out", default=None, help="Output CSV path")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if not args.squad_csv or not args.player_stats:
        parser.print_help()
        return

    squad_df = pd.read_csv(Path(args.squad_csv))
    player_stats_df = pd.read_csv(Path(args.player_stats))
    team_features = aggregate_national_team_stats(squad_df, player_stats_df)

    output_path = (
        Path(args.out)
        if args.out
        else get_project_root() / "data" / "processed" / "national_team_features.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    team_features.to_csv(output_path, index=False)
    logger.info("Saved national team features to %s", output_path)


if __name__ == "__main__":
    main()
