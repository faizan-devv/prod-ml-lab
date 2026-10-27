# Production ML Lab - common commands. `make` (or `make help`) lists them.
SHELL := /bin/bash
.DEFAULT_GOAL := help
COMPOSE := docker compose
AIRFLOW := $(COMPOSE) exec airflow-scheduler airflow
LAB ?=

help: ## show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup / platform
setup: ## one-time: .env, python deps, progress file
	@command -v docker >/dev/null || { echo "docker is required"; exit 1; }
	@command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
	@test -f .env || cp .env.example .env
	@sed -i.bak -e "s|^AIRFLOW_UID=.*|AIRFLOW_UID=$$(id -u)|" -e "s|^LAB_ROOT=.*|LAB_ROOT=$$(pwd)|" .env && rm -f .env.bak
	@grep -q '^LAB_ROOT=' .env || echo "LAB_ROOT=$$(pwd)" >> .env
	uv sync
	@test -f .lab/progress.json || python3 scripts/lab.py start 00 >/dev/null
	@echo "setup done. Next: make up"

up: ## build images and start every service (waits until healthy)
	$(COMPOSE) up -d --build --wait
	@$(MAKE) --no-print-directory urls

down: ## stop services (keeps all data)
	$(COMPOSE) down

ps: ## service status
	$(COMPOSE) ps --format 'table {{.Service}}\t{{.State}}\t{{.Health}}\t{{.Ports}}'

logs: ## follow logs of one service: make logs S=mlflow
	$(COMPOSE) logs -f --tail 100 $(S)

urls: ## print the UIs
	@set -a; source .env; set +a; \
	echo "  MLflow   http://localhost:$${MLFLOW_PORT:-5000}"; \
	echo "  Airflow  http://localhost:$${AIRFLOW_PORT:-8080}   (airflow / airflow)"; \
	echo "  Feast UI make feast-ui -> http://localhost:8888"; \
	echo "  API      make serve    -> http://localhost:8000/docs"

nuke: ## DANGER: delete all containers, volumes and generated data (CONFIRM=yes)
	@[ "$(CONFIRM)" = "yes" ] || { echo "This deletes MLflow runs, models, Airflow history, Feast registry and ./data."; echo "Run: make nuke CONFIRM=yes"; exit 1; }
	$(COMPOSE) down -v
	rm -rf data .lab/progress.json

# ---------------------------------------------------------------- data + features
data: ## generate bootstrap history (2026-01-01 .. 2026-03-31)
	uv run fraudlab generate

features: ## compute daily feature tables for the bootstrap history
	uv run fraudlab build-features

feast-apply: ## register feature definitions (feature_repo/) in the Feast registry
	cd feature_repo && uv run --env-file ../.env feast apply

feast-plan: ## show what `feast apply` would change, without changing it
	cd feature_repo && uv run --env-file ../.env feast plan

feast-ui: ## Feast web UI on :8888
	cd feature_repo && uv run --env-file ../.env feast ui --port 8888

materialize: ## load latest features (as of 2026-03-31) into Redis
	uv run fraudlab materialize --day 2026-03-31

# ---------------------------------------------------------------- serving
serve: ## run the scoring API on :8000 (Lab 09)
	OMP_NUM_THREADS=1 uv run uvicorn fraudlab.serving.app:app --port 8000

# ---------------------------------------------------------------- airflow
airflow: ## run an airflow CLI command: make airflow CMD="dags list"
	$(AIRFLOW) $(CMD)

# ---------------------------------------------------------------- learning workflow
status: ## your progress through the labs
	@python3 scripts/lab.py status

check: ## run the checks for a lab: make check LAB=04
	@[ -n "$(LAB)" ] || { echo "usage: make check LAB=04"; exit 1; }
	@python3 scripts/lab.py check $(LAB)

solution: ## copy a lab's reference solution in (backs up yours): make solution LAB=04
	@[ -n "$(LAB)" ] || { echo "usage: make solution LAB=04"; exit 1; }
	@python3 scripts/lab.py solution $(LAB)

test: ## your own unit tests (Lab 11)
	uv run pytest tests

lint: ## ruff
	uv run ruff check src dags tests

.PHONY: help setup up down ps logs urls nuke data features feast-apply feast-plan feast-ui \
	materialize serve airflow status check solution test lint
