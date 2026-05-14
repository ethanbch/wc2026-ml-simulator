"""Utilities for mapping national team squads to club player stats."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["team", "player"]
OPTIONAL_COLUMNS = ["club", "position", "minutes", "90s", "season", "league"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_player_name(name: str) -> str:
    if pd.isna(name):
        return ""
    return " ".join(str(name).strip().split())


def to_snake_case(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def load_squad_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Squad file not found: {path}")
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required squad columns: {missing}")
    df["player"] = df["player"].apply(normalize_player_name)
    df["team"] = df["team"].apply(normalize_player_name)
    return df


def build_squad_template(teams: Iterable[str], output_path: Path) -> pd.DataFrame:
    rows = []
    for team in teams:
        rows.append(
            {"team": team, "player": "", "club": "", "position": "", "minutes": ""}
        )
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Template saved to %s", output_path)
    return df


def prepare_player_stats(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    df = player_stats_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(part) for part in col if part]) for col in df.columns
        ]
    df = df.rename(columns={col: to_snake_case(col) for col in df.columns})

    player_col_candidates = ["player", "player_name", "name", "player_name_1"]
    for col in player_col_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "player"})
            break
    if "player" not in df.columns:
        raise ValueError("Player column not found in player stats")

    df["player"] = df["player"].apply(normalize_player_name)
    return df


def merge_squad_with_player_stats(
    squad_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    squad_df = squad_df.copy()
    squad_df["player"] = squad_df["player"].apply(normalize_player_name)

    player_stats_df = prepare_player_stats(player_stats_df)

    merged = squad_df.merge(
        player_stats_df, on="player", how="left", suffixes=("", "_fbref")
    )
    missing = merged[merged.isnull().any(axis=1)]["player"].unique().tolist()
    if missing:
        logger.warning("Players not matched in FBref stats: %s", missing[:10])
    return merged


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Squad mapping utilities")
    parser.add_argument("--squad-csv", default=None, help="Path to squad CSV")
    parser.add_argument("--teams", default=None, help="Comma-separated list of teams")
    parser.add_argument("--out", default=None, help="Output path for template")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if args.teams and args.out:
        teams = [t.strip() for t in args.teams.split(",") if t.strip()]
        build_squad_template(teams, Path(args.out))
        return

    if args.squad_csv:
        df = load_squad_csv(Path(args.squad_csv))
        logger.info("Squad loaded: %s rows", len(df))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
