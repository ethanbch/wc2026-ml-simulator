"""Group stage simulation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTCOME_SCORES = {"W": (1, 0), "D": (1, 1), "L": (0, 1)}


def prepare_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace({pd.NA: np.nan}).infer_objects()
    for col in cleaned.select_dtypes(include=["number"]).columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned


def simulate_group_stage_from_probs(
    matches_df: pd.DataFrame,
    probs: np.ndarray,
    label_classes: list[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"group", "home_team", "away_team"}
    missing = required - set(matches_df.columns)
    if missing:
        raise ValueError(f"Missing group stage columns: {missing}")

    if set(label_classes) != {"W", "D", "L"}:
        raise ValueError("Label classes must be exactly {'W','D','L'}")

    if len(probs) != len(matches_df):
        raise ValueError("Probabilities must align with matches rows")

    standings = {}
    results = []

    for idx, row in matches_df.iterrows():
        group = row["group"]
        home = row["home_team"]
        away = row["away_team"]

        proba = probs[idx]
        outcome = rng.choice(label_classes, p=proba)

        home_goals, away_goals = OUTCOME_SCORES[outcome]
        home_points = 3 if outcome == "W" else 1 if outcome == "D" else 0
        away_points = 3 if outcome == "L" else 1 if outcome == "D" else 0

        for team, points, gf, ga in [
            (home, home_points, home_goals, away_goals),
            (away, away_points, away_goals, home_goals),
        ]:
            key = (group, team)
            if key not in standings:
                standings[key] = {
                    "group": group,
                    "team": team,
                    "points": 0,
                    "gf": 0,
                    "ga": 0,
                }
            standings[key]["points"] += points
            standings[key]["gf"] += gf
            standings[key]["ga"] += ga

        results.append(
            {
                "group": group,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome": outcome,
            }
        )

    standings_df = pd.DataFrame(standings.values())
    standings_df["gd"] = standings_df["gf"] - standings_df["ga"]

    standings_df = standings_df.sort_values(
        ["group", "points", "gd", "gf"],
        ascending=[True, False, False, False],
    )
    standings_df["rank"] = standings_df.groupby("group").cumcount() + 1
    results_df = pd.DataFrame(results)
    return standings_df, results_df


def simulate_group_stage(
    matches_df: pd.DataFrame,
    model,
    feature_columns: list[str],
    label_classes: list[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"group", "home_team", "away_team"}
    missing = required - set(matches_df.columns)
    if missing:
        raise ValueError(f"Missing group stage columns: {missing}")

    if set(label_classes) != {"W", "D", "L"}:
        raise ValueError("Label classes must be exactly {'W','D','L'}")

    standings = {}
    results = []

    for _, row in matches_df.iterrows():
        group = row["group"]
        home = row["home_team"]
        away = row["away_team"]

        X = prepare_features_for_model(row[feature_columns].to_frame().T)
        proba = model.predict_proba(X)[0]
        outcome = rng.choice(label_classes, p=proba)

        home_goals, away_goals = OUTCOME_SCORES[outcome]
        home_points = 3 if outcome == "W" else 1 if outcome == "D" else 0
        away_points = 3 if outcome == "L" else 1 if outcome == "D" else 0

        for team, points, gf, ga in [
            (home, home_points, home_goals, away_goals),
            (away, away_points, away_goals, home_goals),
        ]:
            key = (group, team)
            if key not in standings:
                standings[key] = {
                    "group": group,
                    "team": team,
                    "points": 0,
                    "gf": 0,
                    "ga": 0,
                }
            standings[key]["points"] += points
            standings[key]["gf"] += gf
            standings[key]["ga"] += ga

        results.append(
            {
                "group": group,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome": outcome,
            }
        )

    standings_df = pd.DataFrame(standings.values())
    standings_df["gd"] = standings_df["gf"] - standings_df["ga"]

    standings_df = standings_df.sort_values(
        ["group", "points", "gd", "gf"],
        ascending=[True, False, False, False],
    )
    standings_df["rank"] = standings_df.groupby("group").cumcount() + 1
    results_df = pd.DataFrame(results)
    return standings_df, results_df
