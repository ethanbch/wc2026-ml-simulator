"""Tournament simulation orchestrator."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from .group_stage import prepare_features_for_model, simulate_group_stage_from_probs
from .knockout_stage import select_qualifiers, seed_teams, simulate_knockout

logger = logging.getLogger(__name__)


def simulate_tournament(
    model,
    group_matches_df: pd.DataFrame,
    feature_columns: list[str],
    label_classes: list[str],
    build_features_fn,
    n_simulations: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = pd.unique(
        pd.concat([group_matches_df["home_team"], group_matches_df["away_team"]])
    ).tolist()

    stage_order = ["R32", "R16", "QF", "SF", "F", "WIN"]
    counts = {team: {stage: 0 for stage in stage_order} for team in teams}

    group_X = prepare_features_for_model(group_matches_df[feature_columns])
    group_probs = model.predict_proba(group_X)

    for sim in tqdm(range(n_simulations), desc="⚽ Simulation du tournoi"):
        standings_df, _ = simulate_group_stage_from_probs(
            group_matches_df, group_probs, label_classes, rng
        )
        qualified = select_qualifiers(standings_df)
        seeds = seed_teams(qualified)

        winner, stage_reached = simulate_knockout(
            seeds, build_features_fn, model, feature_columns, label_classes, rng
        )

        stage_index = {stage: idx for idx, stage in enumerate(stage_order)}
        for team, stage in stage_reached.items():
            idx = stage_index[stage]
            for stage_name in stage_order[: idx + 1]:
                counts[team][stage_name] += 1

    results = []
    for team, stage_counts in counts.items():
        results.append(
            {
                "team": team,
                "qual_prob": stage_counts["R32"] / n_simulations,
                "r16_prob": stage_counts["R16"] / n_simulations,
                "qf_prob": stage_counts["QF"] / n_simulations,
                "sf_prob": stage_counts["SF"] / n_simulations,
                "final_prob": stage_counts["F"] / n_simulations,
                "win_prob": stage_counts["WIN"] / n_simulations,
            }
        )

    results_df = pd.DataFrame(results).sort_values("win_prob", ascending=False)
    return results_df
