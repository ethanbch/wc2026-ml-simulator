"""Run Monte Carlo simulation for WC 2026."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

from src.data_collection.data_loader import normalize_team_name
from src.features.elo_calculator import compute_latest_elo
from src.features.rolling_form import compute_latest_form_features
from src.simulation.tournament_simulator import simulate_tournament

logger = logging.getLogger(__name__)


def load_json(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_teams(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team_name)
    return df


def load_latest_rankings(rankings_df: pd.DataFrame) -> pd.DataFrame:
    rankings_df = rankings_df.copy()
    column_map = {
        "country_full": "team",
        "country": "team",
        "team_name": "team",
        "total_points": "points",
    }
    for raw_col, new_col in column_map.items():
        if raw_col in rankings_df.columns:
            rankings_df = rankings_df.rename(columns={raw_col: new_col})

    if "team" not in rankings_df.columns:
        raise KeyError(
            "FIFA rankings missing 'team' column. Columns found: "
            f"{rankings_df.columns.tolist()}"
        )

    rankings_df["rank_date"] = pd.to_datetime(rankings_df["rank_date"], errors="coerce")
    latest = rankings_df.sort_values("rank_date").groupby("team").tail(1)
    latest = latest.rename(columns={"rank": "fifa_rank", "points": "fifa_points"})
    return latest[["team", "fifa_rank", "fifa_points", "confederation"]]


def build_team_state(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame | None,
    team_features_df: pd.DataFrame | None,
    form_n: int = 8,
    form_decay: float = 0.85,
) -> pd.DataFrame:
    elo_df = compute_latest_elo(matches_df)
    form_df = compute_latest_form_features(matches_df, n=form_n, decay=form_decay)

    team_state = elo_df.merge(form_df, on="team", how="outer")

    if rankings_df is not None:
        latest_rankings = load_latest_rankings(rankings_df)
        team_state = team_state.merge(latest_rankings, on="team", how="left")

    if team_features_df is not None:
        team_state = team_state.merge(team_features_df, on="team", how="left")

    return team_state.set_index("team")


def build_match_features(
    home_team: str,
    away_team: str,
    team_state: pd.DataFrame,
    feature_columns: list[str],
    stage: str,
    form_n: int,
) -> pd.DataFrame:
    if home_team not in team_state.index or away_team not in team_state.index:
        missing = [
            team for team in [home_team, away_team] if team not in team_state.index
        ]
        raise ValueError(f"Missing team state for: {missing}")

    home = team_state.loc[home_team]
    away = team_state.loc[away_team]

    row = {
        "home_team": home_team,
        "away_team": away_team,
        "elo_home": home.get("elo"),
        "elo_away": away.get("elo"),
        "match_importance": 1.0 if stage.lower().startswith("g") else 1.5,
        "days_rest_home": 7,
        "days_rest_away": 7,
    }
    row["elo_diff"] = row["elo_home"] - row["elo_away"]

    for metric in [
        "goals_scored",
        "goals_conceded",
        "win_rate",
        "clean_sheet_rate",
    ]:
        home_key = f"form_{metric}_{form_n}"
        away_key = f"form_{metric}_{form_n}"
        row[f"home_form_{metric}_{form_n}"] = home.get(home_key)
        row[f"away_form_{metric}_{form_n}"] = away.get(away_key)

    row["home_fifa_rank"] = home.get("fifa_rank")
    row["away_fifa_rank"] = away.get("fifa_rank")
    row["home_fifa_points"] = home.get("fifa_points")
    row["away_fifa_points"] = away.get("fifa_points")
    row["fifa_rank_diff"] = row["home_fifa_rank"] - row["away_fifa_rank"]
    row["fifa_points_diff"] = row["home_fifa_points"] - row["away_fifa_points"]

    row["home_confederation"] = home.get("confederation")
    row["away_confederation"] = away.get("confederation")
    row["same_confederation"] = row["home_confederation"] == row["away_confederation"]

    squad_cols = [col for col in team_state.columns if col.startswith("squad_")]
    for col in squad_cols:
        row[f"home_{col}"] = home.get(col)
        row[f"away_{col}"] = away.get(col)

    feature_df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = pd.NA
    feature_df = feature_df[feature_columns]
    return feature_df


def build_group_match_features(
    schedule_df: pd.DataFrame,
    team_state: pd.DataFrame,
    feature_columns: list[str],
    form_n: int,
) -> pd.DataFrame:
    rows = []
    # On ajoute la barre ici pour la préparation des matchs
    for _, match in tqdm(
        schedule_df.iterrows(),
        total=len(schedule_df),
        desc="🛠️ Building group features",
    ):
        feature_df = build_match_features(
            match["home_team"],
            match["away_team"],
            team_state,
            feature_columns,
            stage="group",
            form_n=form_n,
        )
        row = {
            "group": match["group"],
            "home_team": match["home_team"],
            "away_team": match["away_team"],
        }
        for col in feature_columns:
            row[col] = feature_df.iloc[0][col]
        rows.append(row)
    return pd.DataFrame(rows)


def load_schedule(schedule_path: Path) -> pd.DataFrame:
    df = pd.read_csv(schedule_path)
    df.columns = [col.strip().lower() for col in df.columns]
    rename_map = {
        "home": "home_team",
        "away": "away_team",
        "team1": "home_team",
        "team2": "away_team",
    }
    df = df.rename(columns=rename_map)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Simulate WC 2026")
    parser.add_argument("--model-path", required=True, help="Trained model path")
    parser.add_argument(
        "--models-dir", default="data/processed/models", help="Models dir"
    )
    parser.add_argument("--feature-columns", default=None, help="Feature columns JSON")
    parser.add_argument("--label-classes", default=None, help="Label classes JSON")
    parser.add_argument(
        "--historical-matches", required=True, help="Historical matches CSV"
    )
    parser.add_argument("--fifa-rankings", default=None, help="FIFA rankings CSV")
    parser.add_argument("--team-features", default=None, help="Team features CSV")
    parser.add_argument("--groups-csv", required=True, help="Groups CSV")
    parser.add_argument("--schedule-csv", required=True, help="Schedule CSV")
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--output", default="data/processed/wc2026_simulation.csv")
    parser.add_argument("--form-n", type=int, default=8)
    parser.add_argument("--form-decay", type=float, default=0.85)
    args = parser.parse_args()

    model = load(args.model_path)

    models_dir = Path(args.models_dir)
    feature_columns_path = (
        Path(args.feature_columns)
        if args.feature_columns
        else models_dir / "feature_columns.json"
    )
    label_classes_path = (
        Path(args.label_classes)
        if args.label_classes
        else models_dir / "label_classes.json"
    )
    feature_columns = load_json(feature_columns_path)
    label_classes = load_json(label_classes_path)

    matches_df = pd.read_csv(args.historical_matches)
    matches_df = normalize_teams(matches_df, ["home_team", "away_team"])

    rankings_df = None
    if args.fifa_rankings:
        rankings_df = pd.read_csv(args.fifa_rankings)
        rankings_df = normalize_teams(rankings_df, ["team"])

    team_features_df = None
    if args.team_features:
        team_features_df = pd.read_csv(args.team_features)
        team_features_df = normalize_teams(team_features_df, ["team"])

    team_state = build_team_state(
        matches_df,
        rankings_df,
        team_features_df,
        form_n=args.form_n,
        form_decay=args.form_decay,
    )

    groups_df = pd.read_csv(args.groups_csv)
    groups_df = normalize_teams(groups_df, ["team"])
    schedule_df = load_schedule(Path(args.schedule_csv))
    schedule_df = normalize_teams(schedule_df, ["home_team", "away_team"])

    if "stage" in schedule_df.columns:
        schedule_df = schedule_df[
            schedule_df["stage"].astype(str).str.contains("group", case=False, na=False)
        ].copy()

    if "group" not in schedule_df.columns and "group" in groups_df.columns:
        group_map = groups_df.set_index("team")["group"].to_dict()
        schedule_df["group"] = schedule_df["home_team"].map(group_map)

    if "group" not in schedule_df.columns:
        raise ValueError("Schedule CSV must include a 'group' column")

    group_matches_df = build_group_match_features(
        schedule_df,
        team_state,
        feature_columns,
        form_n=args.form_n,
    )

    def build_features_fn(home: str, away: str, stage: str) -> pd.DataFrame:
        return build_match_features(
            home, away, team_state, feature_columns, stage=stage, form_n=args.form_n
        )

    results_df = simulate_tournament(
        model,
        group_matches_df,
        feature_columns,
        label_classes,
        build_features_fn,
        n_simulations=args.n_simulations,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Saved simulation results to %s", output_path)


if __name__ == "__main__":
    main()
