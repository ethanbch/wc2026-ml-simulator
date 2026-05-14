"""Build the match-level dataset from raw sources."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_collection.data_loader import (
    load_fifa_rankings,
    load_international_results,
)
from src.features.elo_calculator import add_elo_features
from src.features.fbref_aggregator import aggregate_national_team_stats
from src.features.feature_builder import build_match_dataset
from src.features.rolling_form import add_rolling_form_features

from . import config

logger = logging.getLogger(__name__)


def build_dataset(
    international_results: Path = config.INTL_RESULTS,
    fifa_rankings: Path | None = None,
    squad_csv: Path | None = None,
    player_stats: Path | None = None,
    dataset_out: Path = config.MATCH_DATASET,
    team_features_out: Path | None = None,
    elo_out: Path | None = None,
    form_out: Path | None = None,
    min_date: str = "2010-01-01",
    official_only: bool = True,
) -> None:
    matches_df = load_international_results(
        international_results, min_date=min_date, official_only=official_only
    )

    fifa_df = None
    if fifa_rankings:
        fifa_df = load_fifa_rankings(fifa_rankings, min_date=min_date)

    team_features_df = None
    if squad_csv and player_stats:
        squad_df = pd.read_csv(squad_csv)
        player_stats_df = pd.read_csv(player_stats)
        team_features_df = aggregate_national_team_stats(squad_df, player_stats_df)
        if team_features_out:
            team_features_out.parent.mkdir(parents=True, exist_ok=True)
            team_features_df.to_csv(team_features_out, index=False)
            logger.info("Saved team features to %s", team_features_out)

    if elo_out:
        _, history_df = add_elo_features(matches_df, return_history=True)
        elo_out.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(elo_out, index=False)
        logger.info("Saved Elo history to %s", elo_out)

    if form_out:
        form_df = add_rolling_form_features(matches_df)
        form_out.parent.mkdir(parents=True, exist_ok=True)
        form_df.to_csv(form_out, index=False)
        logger.info("Saved rolling form dataset to %s", form_out)

    dataset = build_match_dataset(matches_df, fifa_df, team_features_df)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_out, index=False)
    logger.info("Saved match dataset to %s", dataset_out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build match-level dataset")
    parser.add_argument("--international-results", required=True, help="Results CSV")
    parser.add_argument("--fifa-rankings", default=None, help="FIFA rankings CSV")
    parser.add_argument("--squad-csv", default=None, help="Squad CSV")
    parser.add_argument("--player-stats", default=None, help="Player stats CSV")
    parser.add_argument(
        "--dataset-out",
        default=str(config.MATCH_DATASET),
        help="Output match dataset CSV",
    )
    parser.add_argument(
        "--team-features-out",
        default=str(config.TEAM_FEATURES),
        help="Output team features CSV",
    )
    parser.add_argument(
        "--elo-out",
        default=str(config.ELO_HISTORY),
        help="Output Elo history CSV",
    )
    parser.add_argument(
        "--form-out",
        default=str(config.FORM_FEATURES),
        help="Output rolling form CSV",
    )
    parser.add_argument("--min-date", default="2010-01-01", help="Minimum date")
    parser.add_argument(
        "--include-friendlies",
        action="store_true",
        help="Include friendly matches",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    build_dataset(
        international_results=Path(args.international_results),
        fifa_rankings=Path(args.fifa_rankings) if args.fifa_rankings else None,
        squad_csv=Path(args.squad_csv) if args.squad_csv else None,
        player_stats=Path(args.player_stats) if args.player_stats else None,
        dataset_out=Path(args.dataset_out),
        team_features_out=(
            Path(args.team_features_out) if args.team_features_out else None
        ),
        elo_out=Path(args.elo_out) if args.elo_out else None,
        form_out=Path(args.form_out) if args.form_out else None,
        min_date=args.min_date,
        official_only=not args.include_friendlies,
    )


if __name__ == "__main__":
    main()
