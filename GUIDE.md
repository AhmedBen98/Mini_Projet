# Guide d'Utilisation - Projet MLOps Diabetes

##  Démarrage Rapide

### 1. Installation des dépendances

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Génération des versions du dataset

```bash
# Générer toutes les versions (v1, v2, v3)
python src/generate_diabetes_versions.py --version all

# Ou générer une version spécifique
python src/generate_diabetes_versions.py --version v1
python src/generate_diabetes_versions.py --version v2
python src/generate_diabetes_versions.py --version v3
```

**Résultat**: Trois fichiers créés dans `data/raw/`:
- `diabetes_v1_original.csv` (442 échantillons)
- `diabetes_v2_augmented.csv` (542 échantillons)
- `diabetes_v3_varied.csv` (642 échantillons)

### 3. Initialiser Git et DVC

```bash
# Git
git init
git add .
git commit -m "Initial commit: Diabetes MLOps project"

# DVC
dvc init
dvc remote add -d myremote /tmp/dvc-storage
```

##  Exécution du Pipeline Complet

### Version 1 - Dataset Original (442 samples)

```bash
# 1. Preprocessing
python src/data_processing.py \
  --input data/raw/diabetes_v1_original.csv \
  --output data/processed/v1

# 2. Feature Engineering
python src/feature_engineering.py \
  --train data/processed/v1/train.csv \
  --val data/processed/v1/val.csv \
  --test data/processed/v1/test.csv \
  --output data/processed/v1 \
  --dataset diabetes

# 3. Training
python src/train.py \
  --train data/processed/v1/train_engineered.csv \
  --val data/processed/v1/val_engineered.csv \
  --dataset diabetes \
  --output models/v1

# 4. Evaluation
python src/evaluate.py \
  --test data/processed/v1/test_engineered.csv \
  --model models/v1/model.pkl \
  --dataset diabetes

# 5. Optimization avec Optuna (optionnel)
python src/optimize.py \
  --train data/processed/v1/train_engineered.csv \
  --val data/processed/v1/val_engineered.csv \
  --dataset diabetes \
  --output models/v1

# 6. Versionner avec DVC
dvc add data/raw/diabetes_v1_original.csv
dvc add models/v1/model.pkl
git add data/raw/diabetes_v1_original.csv.dvc models/v1/model.pkl.dvc
git commit -m "Version 1: Original dataset (442 samples)"
git tag v1.0
dvc push
```

### Version 2 - Dataset Augmenté (+100 samples)

```bash
# Créer une branche
git checkout -b dataset-v2

# 1. Preprocessing
python src/data_processing.py \
  --input data/raw/diabetes_v2_augmented.csv \
  --output data/processed/v2

# 2. Feature Engineering
python src/feature_engineering.py \
  --train data/processed/v2/train.csv \
  --val data/processed/v2/val.csv \
  --test data/processed/v2/test.csv \
  --output data/processed/v2 \
  --dataset diabetes

# 3. Training
python src/train.py \
  --train data/processed/v2/train_engineered.csv \
  --val data/processed/v2/val_engineered.csv \
  --dataset diabetes \
  --output models/v2

# 4. Evaluation
python src/evaluate.py \
  --test data/processed/v2/test_engineered.csv \
  --model models/v2/model.pkl \
  --dataset diabetes

# 5. Optimization
python src/optimize.py \
  --train data/processed/v2/train_engineered.csv \
  --val data/processed/v2/val_engineered.csv \
  --dataset diabetes \
  --output models/v2

# 6. Versionner
dvc add data/raw/diabetes_v2_augmented.csv
dvc add models/v2/model.pkl
git add data/raw/diabetes_v2_augmented.csv.dvc models/v2/model.pkl.dvc
git commit -m "Version 2: Augmented dataset (+100 samples = 542)"
git tag v2.0
dvc push
```

### Version 3 - Dataset avec Variations (+200 samples)

```bash
# Créer une branche
git checkout -b dataset-v3

# Pipeline complet (même structure que v2)
python src/data_processing.py --input data/raw/diabetes_v3_varied.csv --output data/processed/v3
python src/feature_engineering.py --train data/processed/v3/train.csv --val data/processed/v3/val.csv --test data/processed/v3/test.csv --output data/processed/v3 --dataset diabetes
python src/train.py --train data/processed/v3/train_engineered.csv --val data/processed/v3/val_engineered.csv --dataset diabetes --output models/v3
python src/evaluate.py --test data/processed/v3/test_engineered.csv --model models/v3/model.pkl --dataset diabetes
python src/optimize.py --train data/processed/v3/train_engineered.csv --val data/processed/v3/val_engineered.csv --dataset diabetes --output models/v3

# Versionner
dvc add data/raw/diabetes_v3_varied.csv
dvc add models/v3/model.pkl
git add data/raw/diabetes_v3_varied.csv.dvc models/v3/model.pkl.dvc
git commit -m "Version 3: Dataset with variations (+200 samples = 642)"
git tag v3.0
dvc push
```

##  Comparaison des Résultats

```bash
# Générer le rapport de comparaison
python src/compare_versions.py \
  --v1-data data/raw/diabetes_v1_original.csv \
  --v2-data data/raw/diabetes_v2_augmented.csv \
  --v3-data data/raw/diabetes_v3_varied.csv \
  --v1-metrics metrics/v1/test_metrics.json \
  --v2-metrics metrics/v2/test_metrics.json \
  --v3-metrics metrics/v3/test_metrics.json \
  --output reports

# Ouvrir le rapport
cat reports/comparison_report.md
```

##  Visualisation MLflow

```bash
# Lancer MLflow UI
mlflow ui --port 5000

# Ouvrir dans le navigateur
# http://localhost:5000
```

Dans l'interface MLflow:
1. Comparer les runs des 3 versions
2. Visualiser les métriques (RMSE, R², MAE)
3. Comparer les hyperparamètres
4. Télécharger les modèles

##  Tests

```bash
# Exécuter tous les tests
pytest tests/ -v

# Avec couverture
pytest tests/ -v --cov=src --cov-report=html

# Ouvrir le rapport de couverture
open htmlcov/index.html
```

##  Pipeline DVC Automatisé

```bash
# Créer le pipeline DVC
dvc repro

# Afficher le DAG
dvc dag

# Comparer les métriques entre versions
dvc metrics show
dvc metrics diff v1.0 v2.0
dvc metrics diff v2.0 v3.0
```

## 🤖 CI/CD avec GitHub Actions

Le workflow `.github/workflows/ml-pipeline.yml` s'exécute automatiquement sur:
- Push sur `main` ou branches `dataset-*`
- Pull requests

Étapes:
1. Tests unitaires
2. Linting du code
3. Validation des données
4. Exécution du pipeline ML
5. Upload des artefacts (métriques, modèles)

##  Structure des Résultats

```
mlops-advanced-project/
├── metrics/
│   ├── v1/
│   │   ├── train_metrics.json
│   │   ├── test_metrics.json
│   │   └── predictions.csv
│   ├── v2/
│   │   └── ...
│   └── v3/
│       └── ...
├── models/
│   ├── v1/
│   │   ├── model.pkl
│   │   └── model_optimized.pkl
│   ├── v2/
│   │   └── ...
│   └── v3/
│       └── ...
├── plots/
│   ├── v1/
│   │   ├── feature_importance.png
│   │   ├── regression_results.png
│   │   └── prediction_distribution.png
│   ├── v2/
│   │   └── ...
│   └── v3/
│       └── ...
└── reports/
    ├── comparison_report.md
    ├── dataset_comparison.json
    ├── target_distributions.png
    ├── feature_distributions.png
    └── model_performance_comparison.png
```

##  Commandes Utiles

```bash
# Voir l'état DVC
dvc status

# Voir les différences de données
dvc diff

# Récupérer une version spécifique
git checkout v1.0
dvc checkout

# Lister les expériences MLflow
mlflow experiments list

# Comparer deux runs MLflow
mlflow runs compare <run_id_1> <run_id_2>

# Voir les logs
tail -f logs/app.log
```

##  Dépannage

### Problème: MLflow ne trouve pas les runs
```bash
# Vérifier le tracking URI
echo $MLFLOW_TRACKING_URI

# Le définir si nécessaire
export MLFLOW_TRACKING_URI=./mlruns
```

### Problème: DVC ne trouve pas le remote
```bash
# Reconfigurer le remote
dvc remote add -d myremote /tmp/dvc-storage
dvc remote list
```

### Problème: Erreur d'import de modules
```bash
# Ajouter le répertoire src au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

##  Ressources

- **Documentation MLflow**: https://mlflow.org/docs/latest/
- **Documentation DVC**: https://dvc.org/doc
- **Documentation Optuna**: https://optuna.readthedocs.io/
- **Dataset Diabetes**: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

##  Checklist du Projet

- [x] Génération de 3 versions du dataset
- [x] Pipeline de preprocessing
- [x] Feature engineering
- [x] Training avec MLflow tracking
- [x] Evaluation et métriques
- [x] Optimisation Optuna
- [x] Versioning Git
- [x] Versioning DVC
- [x] Tests unitaires
- [x] CI/CD GitHub Actions
- [x] Rapport de comparaison
- [x] Documentation complète
