"""Knockout stage simulation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_qualifiers(
    standings_df: pd.DataFrame,
    n_best_third: int = 8,
) -> pd.DataFrame:
    top2 = standings_df[standings_df["rank"] <= 2]
    third = standings_df[standings_df["rank"] == 3]
    third = third.sort_values(["points", "gd", "gf"], ascending=False)
    best_third = third.head(n_best_third)
    qualified = pd.concat([top2, best_third], ignore_index=True)
    return qualified


def seed_teams(qualified_df: pd.DataFrame) -> list[str]:
    seeded = qualified_df.sort_values(["points", "gd", "gf"], ascending=False)
    return seeded["team"].tolist()


def build_round_pairs(teams: list[str]) -> list[tuple[str, str]]:
    if len(teams) % 2 != 0:
        raise ValueError("Number of teams must be even for knockout rounds")
    half = len(teams) // 2
    return list(zip(teams[:half], list(reversed(teams[half:]))))


def simulate_knockout(
    teams: list[str],
    build_features_fn,
    model,
    feature_columns: list[str],
    label_classes: list[str],
    rng: np.random.Generator,
) -> tuple[str, dict[str, str]]:
    if set(label_classes) != {"W", "D", "L"}:
        raise ValueError("Label classes must be exactly {'W','D','L'}")

    stage_order = ["R32", "R16", "QF", "SF", "F", "WIN"]
    stage_reached = {team: "R32" for team in teams}

    current = teams[:]
    for stage_idx, stage_name in enumerate(stage_order[:-1]):
        next_stage = stage_order[stage_idx + 1]
        pairs = build_round_pairs(current)
        winners = []
        for home, away in pairs:
            X = build_features_fn(home, away, stage_name)
            X_input = X[feature_columns].copy()
            X_input = X_input.replace({pd.NA: np.nan}).infer_objects()
            for col in X_input.select_dtypes(include=["number"]).columns:
                X_input[col] = pd.to_numeric(X_input[col], errors="coerce")
            proba = model.predict_proba(X_input)[0]
            outcome = rng.choice(label_classes, p=proba)

            if outcome == "D":
                p_home = proba[label_classes.index("W")]
                p_away = proba[label_classes.index("L")]
                total = p_home + p_away
                if total == 0:
                    winner = rng.choice([home, away])
                else:
                    winner = rng.choice(
                        [home, away], p=[p_home / total, p_away / total]
                    )
            elif outcome == "W":
                winner = home
            else:
                winner = away

            winners.append(winner)
            stage_reached[winner] = next_stage

        current = winners

    winner = current[0]
    stage_reached[winner] = "WIN"
    return winner, stage_reached
