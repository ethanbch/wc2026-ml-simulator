"""Shared path defaults for pipeline entrypoints."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
FBREF_CACHE = DATA_DIR / "fbref_cache"
FBREF_DIR = RAW_DIR / "fbref"

INTL_RESULTS = RAW_DIR / "international_results.csv"
FIFA_RANKINGS = RAW_DIR / "fifa_rankings.csv"
SQUAD_CSV = RAW_DIR / "squads.csv"
PLAYER_STATS = FBREF_DIR / "player_standard.csv"

TEAM_FEATURES = PROC_DIR / "national_team_features.csv"
MATCH_DATASET = PROC_DIR / "match_dataset.csv"
ELO_HISTORY = PROC_DIR / "elo_ratings.csv"
FORM_FEATURES = PROC_DIR / "matches_with_form.csv"
MODELS_DIR = PROC_DIR / "models"
FEATURE_COLUMNS = MODELS_DIR / "feature_columns.json"
LABEL_CLASSES = MODELS_DIR / "label_classes.json"
WC_GROUPS = RAW_DIR / "wc2026_groups.csv"
WC_SCHEDULE = RAW_DIR / "wc2026_schedule.csv"
SIM_OUTPUT = PROC_DIR / "wc2026_simulation.csv"
DEFAULT_N_SIMULATIONS = 1000
