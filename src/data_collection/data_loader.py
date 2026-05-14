"""Load raw datasets from CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def normalize_team_name(name: str) -> str:
    if pd.isna(name):
        return ""
    normalized = " ".join(str(name).strip().split())
    name_map = {
        "Cura\u00e7ao": "Curacao",
    }
    return name_map.get(normalized, normalized)


def load_international_results(
    csv_path: Path,
    min_date: str | None = "2010-01-01",
    official_only: bool = True,
    friendly_keywords: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = safe_read_csv(csv_path)
    required = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in international results: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if min_date:
        df = df[df["date"] >= pd.to_datetime(min_date)]

    for col in ["home_team", "away_team"]:
        df[col] = df[col].apply(normalize_team_name)

    friendly_keywords = friendly_keywords or ["friendly", "exhibition", "test"]
    df["tournament_lower"] = df["tournament"].astype(str).str.lower()
    df["is_friendly"] = df["tournament_lower"].apply(
        lambda val: any(keyword in val for keyword in friendly_keywords)
    )
    if official_only:
        df = df[~df["is_friendly"]]

    if "neutral" not in df.columns:
        df["neutral"] = False

    df = df.drop(columns=["tournament_lower"]).reset_index(drop=True)
    df["match_id"] = df.index.astype(int)
    return df


def load_fifa_rankings(csv_path: Path, min_date: str | None = None) -> pd.DataFrame:
    df = safe_read_csv(csv_path)

    column_map = {
        "rank_date": "rank_date",
        "country_full": "team",
        "rank": "rank",
        "total_points": "points",
        "confederation": "confederation",
    }
    for raw_col, new_col in column_map.items():
        if raw_col in df.columns:
            df = df.rename(columns={raw_col: new_col})

    required = ["rank_date", "team", "rank"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in FIFA rankings: {missing}")

    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    if min_date:
        df = df[df["rank_date"] >= pd.to_datetime(min_date)]

    df["team"] = df["team"].apply(normalize_team_name)
    return df


def load_wc_groups(csv_path: Path) -> pd.DataFrame:
    return safe_read_csv(csv_path)


def load_wc_schedule(csv_path: Path) -> pd.DataFrame:
    return safe_read_csv(csv_path)
