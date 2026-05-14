"""Run the full MLFoot pipeline in the expected order."""

from __future__ import annotations

import argparse
import logging

from . import config
from . import build_dataset as build_dataset_pipeline
from . import evaluate as evaluate_pipeline
from . import simulate_wc2026 as simulate_pipeline
from . import train_models as train_pipeline

logger = logging.getLogger(__name__)


def run_full_pipeline(n_simulations: int = config.DEFAULT_N_SIMULATIONS) -> None:
    logger.info("Step 1/4: build dataset")
    build_dataset_pipeline.build_dataset(
        international_results=config.INTL_RESULTS,
        fifa_rankings=config.FIFA_RANKINGS,
        squad_csv=config.SQUAD_CSV,
        player_stats=config.PLAYER_STATS,
        dataset_out=config.MATCH_DATASET,
        team_features_out=config.TEAM_FEATURES,
        elo_out=config.ELO_HISTORY,
        form_out=config.FORM_FEATURES,
    )

    logger.info("Step 2/4: train models")
    train_pipeline.train_models(
        dataset=config.MATCH_DATASET,
        models_dir=config.MODELS_DIR,
    )

    logger.info("Step 3/4: evaluate models")
    evaluate_pipeline.evaluate_models(
        dataset=config.MATCH_DATASET,
        models_dir=config.MODELS_DIR,
        output=config.PROC_DIR / "metrics.csv",
    )

    logger.info("Step 4/4: simulate tournament")
    simulate_pipeline.simulate_wc2026(
        model_path=config.MODELS_DIR / "champion_xgboost.pkl",
        models_dir=config.MODELS_DIR,
        feature_columns=config.FEATURE_COLUMNS,
        label_classes=config.LABEL_CLASSES,
        historical_matches=config.INTL_RESULTS,
        fifa_rankings=config.FIFA_RANKINGS,
        team_features=config.TEAM_FEATURES,
        groups_csv=config.WC_GROUPS,
        schedule_csv=config.WC_SCHEDULE,
        output=config.SIM_OUTPUT,
        n_simulations=n_simulations,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=config.DEFAULT_N_SIMULATIONS,
        help="Number of Monte Carlo simulations",
    )
    args = parser.parse_args(argv)
    run_full_pipeline(n_simulations=args.n_simulations)


if __name__ == "__main__":
    main()
