# Makefile pour le Projet MLOps Diabetes

.PHONY: help install setup generate-data run-v1 run-v2 run-v3 run-all compare test clean mlflow-ui demo

# Variables
PYTHON := python3
PIP := pip
VENV := venv

help:
	@echo "Commandes disponibles:"
	@echo "  make install       - Installer les dépendances"
	@echo "  make setup         - Setup complet (venv + install + git/dvc)"
	@echo "  make generate-data - Générer les 3 versions du dataset"
	@echo "  make run-v1        - Exécuter le pipeline pour Version 1"
	@echo "  make run-v2        - Exécuter le pipeline pour Version 2"
	@echo "  make run-v3        - Exécuter le pipeline pour Version 3"
	@echo "  make run-all       - Exécuter le pipeline pour toutes les versions"
	@echo "  make compare       - Générer le rapport de comparaison"
	@echo "  make demo          - Exécuter la démo automatique complète"
	@echo "  make test          - Exécuter les tests"
	@echo "  make mlflow-ui     - Lancer MLflow UI"
	@echo "  make clean         - Nettoyer les fichiers générés"

install:
	@echo "Installation des dépendances..."
	$(PIP) install -r requirements.txt

setup:
	@echo "Setup complet du projet..."
	python -m venv $(VENV)
	@echo "Activez le venv avec: source $(VENV)/bin/activate"
	@echo "Puis exécutez: make install"

init-git-dvc:
	@echo "Initialisation Git et DVC..."
	git init || true
	git config user.name "MLOps User" || true
	git config user.email "mlops@example.com" || true
	dvc init || true
	dvc remote add -d myremote /tmp/dvc-storage || true

generate-data:
	@echo "Génération des 3 versions du dataset..."
	$(PYTHON) src/generate_diabetes_versions.py --version all

run-v1:
	@echo "Exécution du pipeline - Version 1 (Original 442)..."
	mkdir -p data/processed/v1 models/v1 metrics/v1 plots/v1
	PYTHONPATH=. $(PYTHON) src/data_processing.py --input data/raw/diabetes_v1_original.csv --output data/processed/v1
	PYTHONPATH=. $(PYTHON) src/feature_engineering.py --train data/processed/v1/train.csv --val data/processed/v1/val.csv --test data/processed/v1/test.csv --output data/processed/v1 --dataset diabetes
	PYTHONPATH=. $(PYTHON) src/train.py --train data/processed/v1/train_engineered.csv --val data/processed/v1/val_engineered.csv --dataset diabetes --output models/v1
	PYTHONPATH=. $(PYTHON) src/evaluate.py --test data/processed/v1/test_engineered.csv --model models/v1/model.pkl --dataset diabetes --output v1
	cp metrics/train_metrics.json metrics/v1/
	@echo "✅ Version 1 terminée!"

run-v2:
	@echo "Exécution du pipeline - Version 2 (Augmentée +100)..."
	mkdir -p data/processed/v2 models/v2 metrics/v2 plots/v2
	PYTHONPATH=. $(PYTHON) src/data_processing.py --input data/raw/diabetes_v2_augmented.csv --output data/processed/v2
	PYTHONPATH=. $(PYTHON) src/feature_engineering.py --train data/processed/v2/train.csv --val data/processed/v2/val.csv --test data/processed/v2/test.csv --output data/processed/v2 --dataset diabetes
	PYTHONPATH=. $(PYTHON) src/train.py --train data/processed/v2/train_engineered.csv --val data/processed/v2/val_engineered.csv --dataset diabetes --output models/v2
	PYTHONPATH=. $(PYTHON) src/evaluate.py --test data/processed/v2/test_engineered.csv --model models/v2/model.pkl --dataset diabetes --output v2
	cp metrics/train_metrics.json metrics/v2/
	@echo "✅ Version 2 terminée!"

run-v3:
	@echo "Exécution du pipeline - Version 3 (Variée +200)..."
	mkdir -p data/processed/v3 models/v3 metrics/v3 plots/v3
	PYTHONPATH=. $(PYTHON) src/data_processing.py --input data/raw/diabetes_v3_varied.csv --output data/processed/v3
	PYTHONPATH=. $(PYTHON) src/feature_engineering.py --train data/processed/v3/train.csv --val data/processed/v3/val.csv --test data/processed/v3/test.csv --output data/processed/v3 --dataset diabetes
	PYTHONPATH=. $(PYTHON) src/train.py --train data/processed/v3/train_engineered.csv --val data/processed/v3/val_engineered.csv --dataset diabetes --output models/v3
	PYTHONPATH=. $(PYTHON) src/evaluate.py --test data/processed/v3/test_engineered.csv --model models/v3/model.pkl --dataset diabetes --output v3
	cp metrics/train_metrics.json metrics/v3/
	@echo "✅ Version 3 terminée!"
	cp metrics/train_metrics.json metrics/v3/
	cp metrics/test_metrics.json metrics/v3/
	@echo "✅ Version 3 terminée!"

run-all: run-v1 run-v2 run-v3
	@echo "✅ Toutes les versions exécutées!"

compare:
	@echo "Génération du rapport de comparaison..."
	$(PYTHON) src/compare_versions.py \
		--v1-data data/raw/diabetes_v1_original.csv \
		--v2-data data/raw/diabetes_v2_augmented.csv \
		--v3-data data/raw/diabetes_v3_varied.csv \
		--v1-metrics metrics/v1/test_metrics.json \
		--v2-metrics metrics/v2/test_metrics.json \
		--v3-metrics metrics/v3/test_metrics.json \
		--output reports
	@echo "✅ Rapport disponible dans reports/"

demo:
	@echo "Exécution de la démonstration automatique complète..."
	$(PYTHON) run_demo.py

optimize-v1:
	@echo "Optimisation Optuna - Version 1..."
	$(PYTHON) src/optimize.py --train data/processed/v1/train_engineered.csv --val data/processed/v1/val_engineered.csv --dataset diabetes --output models/v1

optimize-v2:
	@echo "Optimisation Optuna - Version 2..."
	$(PYTHON) src/optimize.py --train data/processed/v2/train_engineered.csv --val data/processed/v2/val_engineered.csv --dataset diabetes --output models/v2

optimize-v3:
	@echo "Optimisation Optuna - Version 3..."
	$(PYTHON) src/optimize.py --train data/processed/v3/train_engineered.csv --val data/processed/v3/val_engineered.csv --dataset diabetes --output models/v3

test:
	@echo "Exécution des tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests terminés! Rapport dans htmlcov/index.html"

mlflow-ui:
	@echo "Lancement de MLflow UI..."
	@echo "Ouvrez http://localhost:5000 dans votre navigateur"
	mlflow ui --port 5000

clean:
	@echo "Nettoyage des fichiers générés..."
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf metrics/*
	rm -rf plots/*
	rm -rf reports/*
	rm -rf mlruns/*
	rm -rf logs/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "✅ Nettoyage terminé!"

# Workflow complet
full-workflow: generate-data init-git-dvc run-all compare
	@echo "✅ Workflow complet terminé!"
	@echo "Voir les résultats dans reports/comparison_report.md"
