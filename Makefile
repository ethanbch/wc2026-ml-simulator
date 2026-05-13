PY ?= python

DATA_DIR ?= data
RAW_DIR ?= $(DATA_DIR)/raw
PROC_DIR ?= $(DATA_DIR)/processed
FBREF_CACHE ?= $(DATA_DIR)/fbref_cache
FBREF_DIR ?= $(RAW_DIR)/fbref

LEAGUES ?= Big 5 European Leagues Combined
SEASONS ?= 2022-2024

INTL_RESULTS ?= $(RAW_DIR)/international_results.csv
FIFA_RANKINGS ?= $(RAW_DIR)/fifa_rankings.csv
SQUAD_CSV ?= $(RAW_DIR)/squads.csv
PLAYER_STATS ?= $(FBREF_DIR)/player_standard.csv

TEAM_FEATURES ?= $(PROC_DIR)/national_team_features.csv
MATCH_DATASET ?= $(PROC_DIR)/match_dataset.csv
ELO_HISTORY ?= $(PROC_DIR)/elo_ratings.csv
FORM_FEATURES ?= $(PROC_DIR)/matches_with_form.csv
MODELS_DIR ?= $(PROC_DIR)/models
FEATURE_COLUMNS ?= $(MODELS_DIR)/feature_columns.json
LABEL_CLASSES ?= $(MODELS_DIR)/label_classes.json
WC_GROUPS ?= $(RAW_DIR)/wc2026_groups.csv
WC_SCHEDULE ?= $(RAW_DIR)/wc2026_schedule.csv
SIM_OUTPUT ?= $(PROC_DIR)/wc2026_simulation.csv
N_SIMULATIONS ?= 1000

WITH_TEAM_FEATURES ?= 1

MATCH_ARGS = --matches-csv $(INTL_RESULTS) --fifa-rankings $(FIFA_RANKINGS) --out $(MATCH_DATASET)
ifeq ($(WITH_TEAM_FEATURES),1)
MATCH_ARGS += --team-features $(TEAM_FEATURES)
endif

.PHONY: help fbref team-features elo form match-dataset build-dataset train evaluate simulate dataset full clean

help:
	@echo "Targets:"
	@echo "  fbref             Fetch FBref data via soccerdata"
	@echo "  team-features     Aggregate squad and player stats"
	@echo "  elo               Compute Elo history"
	@echo "  form              Compute rolling form features"
	@echo "  match-dataset     Build match-level dataset"
	@echo "  build-dataset     Full dataset pipeline"
	@echo "  train             Train champion + challengers"
	@echo "  evaluate          Evaluate trained models"
	@echo "  simulate          Run WC2026 simulation"
	@echo "  dataset           Run team-features and match-dataset"
	@echo "  full              Run build-dataset, train, evaluate, simulate"
	@echo "  clean             Remove processed artifacts"

fbref:
	$(PY) -m src.data_collection.fbref_scraper \
		--leagues "$(LEAGUES)" \
		--seasons "$(SEASONS)" \
		--cache-dir "$(FBREF_CACHE)" \
		--output-dir "$(FBREF_DIR)"

team-features:
	$(PY) -m src.features.fbref_aggregator \
		--squad-csv "$(SQUAD_CSV)" \
		--player-stats "$(PLAYER_STATS)" \
		--out "$(TEAM_FEATURES)"

elo:
	$(PY) -m src.features.elo_calculator \
		--matches-csv "$(INTL_RESULTS)" \
		--out "$(ELO_HISTORY)"

form:
	$(PY) -m src.features.rolling_form \
		--matches-csv "$(INTL_RESULTS)" \
		--out "$(FORM_FEATURES)"

match-dataset:
	$(PY) -m src.features.feature_builder $(MATCH_ARGS)

build-dataset:
	$(PY) build_dataset.py \
		--international-results "$(INTL_RESULTS)" \
		--fifa-rankings "$(FIFA_RANKINGS)" \
		--squad-csv "$(SQUAD_CSV)" \
		--player-stats "$(PLAYER_STATS)" \
		--dataset-out "$(MATCH_DATASET)" \
		--team-features-out "$(TEAM_FEATURES)" \
		--elo-out "$(ELO_HISTORY)" \
		--form-out "$(FORM_FEATURES)"

train:
	$(PY) train_models.py \
		--dataset "$(MATCH_DATASET)" \
		--models-dir "$(MODELS_DIR)"

evaluate:
	$(PY) evaluate.py \
		--dataset "$(MATCH_DATASET)" \
		--models-dir "$(MODELS_DIR)"

simulate:
	$(PY) simulate_wc2026.py \
		--model-path "$(MODELS_DIR)/champion_xgboost.pkl" \
		--feature-columns "$(FEATURE_COLUMNS)" \
		--label-classes "$(LABEL_CLASSES)" \
		--historical-matches "$(INTL_RESULTS)" \
		--fifa-rankings "$(FIFA_RANKINGS)" \
		--team-features "$(TEAM_FEATURES)" \
		--groups-csv "$(WC_GROUPS)" \
		--schedule-csv "$(WC_SCHEDULE)" \
		--output "$(SIM_OUTPUT)" \
		--n-simulations "$(N_SIMULATIONS)"

dataset: team-features match-dataset

full: build-dataset train evaluate simulate

clean:
	rm -f "$(TEAM_FEATURES)" "$(MATCH_DATASET)" "$(ELO_HISTORY)" "$(FORM_FEATURES)" "$(SIM_OUTPUT)"
