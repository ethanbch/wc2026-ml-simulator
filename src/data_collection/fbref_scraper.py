"""FBref data collection via soccerdata."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import soccerdata as sd
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "soccerdata is required. Install with `pip install soccerdata`."
    ) from exc

DEFAULT_TEAM_STAT_TYPES = ["standard", "keeper", "shooting", "playing_time", "misc"]
DEFAULT_PLAYER_STAT_TYPE = "standard"
DEFAULT_LEAGUES = "Big 5 European Leagues Combined"
DEFAULT_SEASONS = [2022, 2023, 2024, 2025]

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_seasons(seasons_str: str | None) -> list[int]:
    if not seasons_str:
        return DEFAULT_SEASONS
    seasons_str = seasons_str.strip()
    if "-" in seasons_str and seasons_str.count("-") == 1:
        start_s, end_s = seasons_str.split("-")
        return list(range(int(start_s), int(end_s) + 1))
    return [int(s.strip()) for s in seasons_str.split(",") if s.strip()]


def build_fbref_client(
    leagues: str | Iterable[str],
    seasons: Iterable[int],
    cache_dir: Path | None = None,
    no_cache: bool = False,
    headless: bool = True,
) -> sd.FBref:
    kwargs = {"leagues": leagues, "seasons": seasons}
    if cache_dir is not None:
        kwargs["data_dir"] = cache_dir  # <-- Removed str()
    try:
        return sd.FBref(**kwargs, no_cache=no_cache, headless=headless)
    except TypeError:
        return sd.FBref(**kwargs)


def safe_read(
    read_func, retries: int = 3, retry_wait: int = 5, **kwargs
) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return read_func(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime errors from network
            last_exc = exc
            logger.warning("Read failed (attempt %s/%s): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(retry_wait * attempt)
    raise RuntimeError("FBref read failed after retries") from last_exc


def fetch_team_stats(
    fbref: sd.FBref,
    stat_types: Iterable[str] = DEFAULT_TEAM_STAT_TYPES,
    retries: int = 3,
    retry_wait: int = 5,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for stat_type in stat_types:
        logger.info("Fetching team stats: %s", stat_type)
        df = safe_read(
            fbref.read_team_season_stats,
            stat_type=stat_type,
            retries=retries,
            retry_wait=retry_wait,
        )
        results[stat_type] = df
    return results


def fetch_player_stats(
    fbref: sd.FBref,
    stat_type: str = DEFAULT_PLAYER_STAT_TYPE,
    retries: int = 3,
    retry_wait: int = 5,
) -> pd.DataFrame:
    logger.info("Fetching player stats: %s", stat_type)
    return safe_read(
        fbref.read_player_season_stats,
        stat_type=stat_type,
        retries=retries,
        retry_wait=retry_wait,
    )


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(part) for part in col if part]) for col in df.columns
        ]
    return df


def prepare_export_dataframe(df: pd.DataFrame, is_player: bool) -> pd.DataFrame:
    df = flatten_columns(df)

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

        # On vérifie si la colonne "player" (générée par soccerdata) existe,
        # et on la renomme avec une majuscule si tu le souhaites.
        if is_player and "player" in df.columns:
            df = df.rename(columns={"player": "Player"})

    return df


def save_dataframe(
    df: pd.DataFrame, output_path: Path, *, is_player: bool = False
) -> None:
    ensure_dir(output_path.parent)
    export_df = prepare_export_dataframe(df, is_player=is_player)
    export_df.to_csv(output_path, index=False)
    logger.info("Saved %s rows to %s", len(export_df), output_path)


def save_stats(
    team_stats: dict[str, pd.DataFrame],
    player_stats: pd.DataFrame,
    output_dir: Path,
    player_stat_type: str,
) -> None:
    for stat_type, df in team_stats.items():
        save_dataframe(df, output_dir / f"team_{stat_type}.csv")
    save_dataframe(
        player_stats,
        output_dir / f"player_{player_stat_type}.csv",
        is_player=True,
    )


def run_fbref_pipeline(
    leagues: str | Iterable[str] = DEFAULT_LEAGUES,
    seasons: Iterable[int] = DEFAULT_SEASONS,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    stat_types: Iterable[str] = DEFAULT_TEAM_STAT_TYPES,
    player_stat_type: str = DEFAULT_PLAYER_STAT_TYPE,
    no_cache: bool = False,
    headless: bool = True,
) -> None:
    cache_dir = cache_dir or (get_project_root() / "data" / "fbref_cache")
    output_dir = output_dir or (get_project_root() / "data" / "raw" / "fbref")

    fbref = build_fbref_client(
        leagues=leagues,
        seasons=seasons,
        cache_dir=cache_dir,
        no_cache=no_cache,
        headless=headless,
    )
    team_stats = fetch_team_stats(fbref, stat_types=stat_types)
    player_stats = fetch_player_stats(fbref, stat_type=player_stat_type)
    save_stats(team_stats, player_stats, output_dir, player_stat_type)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch FBref data via soccerdata")
    parser.add_argument("--leagues", default=DEFAULT_LEAGUES, help="FBref league set")
    parser.add_argument(
        "--seasons", default=None, help="Seasons list or range, e.g. 2022-2024"
    )
    parser.add_argument("--cache-dir", default=None, help="Cache directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache usage")
    parser.add_argument("--headless", action="store_true", help="Run headless browser")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_fbref_pipeline(
        leagues=args.leagues,
        seasons=seasons,
        cache_dir=cache_dir,
        output_dir=output_dir,
        no_cache=args.no_cache,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
