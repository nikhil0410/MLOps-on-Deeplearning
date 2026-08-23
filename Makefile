# ---------------------------------------------------------------------------
# Developer convenience targets. Run `make help` to list them.
# ---------------------------------------------------------------------------
.PHONY: help venv install data preprocess train dvc-repro test serve \
        docker-build docker-run compose-up compose-down \
        k8s-deploy k8s-delete smoke simulate clean

PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
IMAGE ?= cats-vs-dogs:latest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtualenv + install dev/training dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
	$(BIN)/pip install -r requirements-dev.txt

data: ## Generate synthetic dataset
	$(BIN)/python -m src.data.generate_synthetic

preprocess: ## Split + resize into train/val/test
	$(BIN)/python -m src.data.preprocess

train: ## Train model + log to MLflow
	$(BIN)/python -m src.models.train

dvc-repro: ## Reproduce the full DVC pipeline
	$(BIN)/dvc repro

test: ## Run unit tests
	$(BIN)/pytest -v

serve: ## Run the API locally (reload)
	$(BIN)/uvicorn src.api.main:app --reload --port 8000

docker-build: ## Build the Docker image
	docker build -t $(IMAGE) .

docker-run: ## Run the container locally
	docker run --rm -p 8000:8000 $(IMAGE)

compose-up: ## Start service + Prometheus via docker compose
	docker compose up --build -d

compose-down: ## Stop docker compose stack
	docker compose down

k8s-deploy: ## Deploy to the current kube context
	kubectl apply -k k8s/
	kubectl rollout status deployment/cats-vs-dogs --timeout=180s

k8s-delete: ## Remove the k8s deployment
	kubectl delete -k k8s/

smoke: ## Run smoke test against localhost:8000
	$(BIN)/python scripts/smoke_test.py --url http://localhost:8000

simulate: ## Send labeled requests + write performance report
	$(BIN)/python scripts/simulate_requests.py --url http://localhost:8000 --n 40

clean: ## Remove generated data, models, caches
	rm -rf data/raw data/processed models/*.pt models/metrics.json models/plots \
	       mlruns .pytest_cache **/__pycache__
