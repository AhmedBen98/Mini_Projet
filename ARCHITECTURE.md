# Architecture du Projet MLOps Avancé - Diabetes Prediction

##  Vue d'ensemble du Projet

Ce projet implémente un pipeline MLOps complet et professionnel pour la prédiction de la progression du diabète, démontrant l'impact de l'augmentation de données sur les performances d'un modèle ML.

### Objectifs Principaux

1.  Construire un pipeline ML reproductible et automatisé
2.  Comparer les performances sur **3 versions du dataset** Diabetes
3.  Intégrer des outils modernes de MLOps (Git, DVC, MLflow, Optuna)
4.  Implémenter des fonctionnalités avancées (Hyperparameter Tuning)
5.  Automatiser le workflow complet avec CI/CD

---

##  Dataset et Modèle

### Diabetes Dataset - 3 Versions

#### Version 1: Original (Baseline)
- **Source**: Scikit-learn built-in dataset
- **Taille**: 442 échantillons
- **Features**: 10 variables biomédicales
  - `age`: Âge
  - `sex`: Sexe
  - `bmi`: Body Mass Index (Indice de masse corporelle)
  - `bp`: Blood Pressure (Pression artérielle)
  - `s1` à `s6`: 6 mesures sériques (blood serum measurements)
- **Target**: Mesure quantitative de la progression du diabète après 1 an
- **Type de problème**: Régression
- **Modèle**: Gradient Boosting Regressor

#### Version 2: Augmenté (+100 samples)
- **Taille**: 542 échantillons (442 + 100 synthétiques)
- **Méthode**: Génération de 100 échantillons avec variation moyenne (15%)
- **Objectif**: Tester l'impact de plus de données

#### Version 3: Avec Variations (+200 samples)
- **Taille**: 642 échantillons (442 + 200 synthétiques)
- **Méthode**: Génération avec 3 niveaux de variation
  - 66 samples avec faible variation (5%)
  - 66 samples avec variation moyenne (15%)
  - 68 samples avec forte variation (30%)
- **Objectif**: Tester la robustesse face à la diversité

---

##  Architecture du Pipeline

### Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                     WORKFLOW MLOps COMPLET                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. GÉNÉRATION DES DONNÉES (generate_diabetes_versions.py)      │
│     - Version 1: 442 samples (original)                         │
│     - Version 2: 542 samples (+100 augmentation)                │
│     - Version 3: 642 samples (+200 variations)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. VERSIONING DVC                                              │
│     - dvc add data/raw/diabetes_v*.csv                          │
│     - Git commit des .dvc files                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. PREPROCESSING (data_processing.py)                          │
│     - Nettoyage des données                                     │
│     - Normalisation (StandardScaler)                            │
│     - Split: Train (70%) / Val (15%) / Test (15%)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. FEATURE ENGINEERING (feature_engineering.py)                │
│     - bmi_age_interaction = bmi * age                           │
│     - bp_bmi_ratio = bp / (bmi + ε)                             │
│     - Sélection des meilleures features                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. TRAINING (train.py) + MLflow Tracking                       │
│     - Gradient Boosting Regressor                               │
│     - Log: params, metrics, artifacts                           │
│     - Validation croisée 5-fold                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. HYPERPARAMETER OPTIMIZATION (optimize.py) + Optuna          │
│     - 50 trials avec TPE Sampler                                │
│     - Pruning: MedianPruner                                     │
│     - Nested MLflow runs pour chaque trial                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. EVALUATION (evaluate.py)                                    │
│     - Métriques: RMSE, MAE, R², MAPE                            │
│     - Visualisations: scatter, residual, importance             │
│     - Sauvegarde des résultats                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. VERSIONING MODÈLES                                          │
│     - dvc add models/v*/model.pkl                               │
│     - Git commit + tag version                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  9. COMPARAISON (compare_versions.py)                           │
│     - Rapport markdown automatique                              │
│     - Comparaison des 3 versions                                │
│     - Visualisations comparatives                               │
└─────────────────────────────────────────────────────────────────┘
```

---

##  Outils et Technologies

### 1. **Git** - Version Control

**Rôle** : Versioning du code source

**Workflow** :
```bash
# Structure des branches
main
├── dataset-v1  # Version 1 (442 samples)
├── dataset-v2  # Version 2 (542 samples)
└── dataset-v3  # Version 3 (642 samples)
```

**Utilisation** :
```bash
git checkout -b dataset-v1
git add .
git commit -m "Add V1 pipeline with 442 samples - R²: 0.4789"
git tag v1.0.0
```

### 2. **DVC** - Data Version Control

**Rôle** : Versioning des données et modèles

**Workflow** :
```bash
# Initialisation
dvc init

# Versioning des données
dvc add data/raw/diabetes_v1_original.csv
git add data/raw/diabetes_v1_original.csv.dvc data/raw/.gitignore
git commit -m "Track V1 dataset with DVC"

# Versioning des modèles
dvc add models/v1/model.pkl
git add models/v1/model.pkl.dvc
git commit -m "Track V1 model"

# Push vers remote storage
dvc push
```

**Avantages** :
- Ne stocke pas les gros fichiers dans Git
- Permet de revenir à n'importe quelle version de données
- Facilite la collaboration

### 3. **MLflow** - Experiment Tracking

**Rôle** : Tracking de toutes les expérimentations

**Configuration** :
```yaml
mlflow:
  tracking_uri: "./mlruns"
  experiment_name: "diabetes-progression-experiment"
  log_models: true
  log_artifacts: true
```

**Métriques Trackées** :
- Hyperparamètres du modèle
- Métriques de performance (RMSE, MAE, R², MAPE)
- Temps d'exécution
- Taille du dataset
- Artifacts (modèles, plots, scalers)

**Utilisation** :
```bash
# Lancer l'UI
mlflow ui --port 5000

# Accès : http://localhost:5000
```

### 4. **Optuna** - Hyperparameter Tuning

**Rôle** : Optimisation automatique des hyperparamètres

**Configuration** :
```yaml
optuna:
  n_trials: 50
  timeout: 3600  # 1 hour
  sampler: "TPE"
  pruner: "MedianPruner"
  direction: "maximize"  # Maximize negative RMSE
```

**Espace de Recherche** :
```python
{
  'n_estimators': [50, 300],
  'learning_rate': [0.01, 0.3],
  'max_depth': [3, 10],
  'min_samples_split': [2, 20],
  'min_samples_leaf': [1, 10],
  'subsample': [0.6, 1.0]
}
```

### 5. **GitHub Actions** - CI/CD

**Rôle** : Automatisation des tests et du pipeline

**Workflow** :
1. Tests unitaires (pytest)
2. Linting (flake8, black)
3. Validation des données
4. Exécution du pipeline ML
5. Upload des artifacts

---

##  Structure du Projet

```
mlops-advanced-project/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml           # CI/CD configuration
│
├── data/
│   ├── raw/                          # Données brutes (DVC tracked)
│   │   ├── diabetes_v1_original.csv
│   │   ├── diabetes_v2_augmented.csv
│   │   └── diabetes_v3_varied.csv
│   └── processed/                    # Données prétraitées
│       ├── v1/, v2/, v3/
│       │   ├── train.csv
│       │   ├── val.csv
│       │   ├── test.csv
│       │   ├── train_engineered.csv
│       │   ├── val_engineered.csv
│       │   ├── test_engineered.csv
│       │   └── scaler.pkl
│
├── models/                           # Modèles (DVC tracked)
│   ├── v1/
│   │   ├── model.pkl
│   │   └── model.pkl.dvc
│   ├── v2/
│   └── v3/
│
├── metrics/                          # Métriques d'évaluation
│   ├── v1/, v2/, v3/
│   │   ├── train_metrics.json
│   │   └── test_metrics.json
│
├── plots/                            # Visualisations
│   ├── v1/, v2/, v3/
│   │   ├── scatter_plot.png
│   │   ├── residual_plot.png
│   │   ├── feature_importance.png
│   │   └── learning_curves.png
│
├── reports/                          # Rapports de comparaison
│   ├── comparison_report.md
│   └── comparison_plots/
│
├── src/                              # Code source
│   ├── __init__.py
│   ├── utils.py                      # Fonctions utilitaires
│   ├── generate_diabetes_versions.py # Générateur de datasets
│   ├── data_processing.py            # Preprocessing
│   ├── feature_engineering.py        # Feature engineering
│   ├── train.py                      # Training + MLflow
│   ├── evaluate.py                   # Evaluation
│   ├── optimize.py                   # Optuna optimization
│   └── compare_versions.py           # Comparaison des versions
│
├── tests/                            # Tests unitaires
│   ├── __init__.py
│   ├── test_utils.py
│   └── test_data_processing.py
│
├── logs/                             # Logs d'exécution
│
├── .gitignore                        # Fichiers à ignorer
├── .dvc/                             # Configuration DVC
├── dvc.yaml                          # Pipeline DVC
├── params.yaml                       # Hyperparamètres
├── requirements.txt                  # Dépendances Python
├── Makefile                          # Commandes automatisées
├── run_demo.py                       # Script de démo automatique
├── README.md                         # Documentation principale
├── ARCHITECTURE.md                   # Ce fichier
├── GUIDE.md                          # Guide d'utilisation
└── RESULTS.md                        # Résultats détaillés
```

---

##  Flux de Données

### Étape 1 : Génération des Datasets

```python
# generate_diabetes_versions.py
class DiabetesDataGenerator:
    def create_version_1_original()  # 442 samples
    def create_version_2_augmented()  # +100 samples (15% variation)
    def create_version_3_varied()     # +200 samples (5%, 15%, 30% variations)
```

### Étape 2 : Preprocessing

```python
# data_processing.py
class DataProcessor:
    def handle_missing_values()      # Imputation
    def remove_outliers()             # IQR method
    def scale_features()              # StandardScaler
    def split_data()                  # Train/Val/Test
```

### Étape 3 : Feature Engineering

```python
# feature_engineering.py
class FeatureEngineer:
    def create_domain_features():
        # bmi_age_interaction = bmi * age
        # bp_bmi_ratio = bp / (bmi + ε)
    
    def select_features():           # Feature selection
```

### Étape 4 : Training

```python
# train.py
class ModelTrainer:
    def get_model():                 # GradientBoostingRegressor
    def train():                     # Training + MLflow logging
    def cross_validate()             # 5-fold CV
```

### Étape 5 : Optimization

```python
# optimize.py
class OptunaOptimizer:
    def objective_diabetes():        # Objective function
    def optimize():                  # Run Optuna study
```

---

##  Métriques d'Évaluation

### Pour Régression (Diabetes)

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **RMSE** | √(Σ(y - ŷ)² / n) | Erreur quadratique moyenne (plus bas = mieux) |
| **MAE** | Σ\|y - ŷ\| / n | Erreur absolue moyenne (plus bas = mieux) |
| **R²** | 1 - (SS_res / SS_tot) | Coefficient de détermination (plus haut = mieux, max=1) |
| **MAPE** | (Σ\|y - ŷ\|/y) / n × 100 | Erreur en pourcentage (plus bas = mieux) |

---

##  Stratégie de Branching Git

```
main (production)
├── dataset-v1
│   └── feature/improve-preprocessing
│   └── feature/add-new-features
│
├── dataset-v2
│   └── experiment/augmentation-strategy
│
└── dataset-v3
    └── experiment/variation-levels
```

### Conventions de Nommage

- `dataset-vX` : Branches principales par version
- `feature/*` : Nouvelles fonctionnalités
- `experiment/*` : Expérimentations
- `fix/*` : Corrections de bugs

---

##  Comparaison des 3 Versions

| Aspect | Version 1 (442) | Version 2 (542) | Version 3 (642) |
|--------|----------------|----------------|----------------|
| **Samples** | 442 | 542 (+100) | 642 (+200) |
| **RMSE** | 52.87 | 50.94 | 49.12 |
| **R²** | 0.4789 | 0.5021 | 0.5234 |
| **Training Time** | 2.15s | 2.34s | 2.67s |
| **Optim Time** | 287s | 312s | 346s |
| **Amélioration** | Baseline | +4.8% | +9.3% |

---

##  Commandes Essentielles

### Setup Initial
```bash
# Installation
pip install -r requirements.txt

# Générer les 3 versions
PYTHONPATH=. python3 src/generate_diabetes_versions.py --version all

# Initialiser Git et DVC
git init
dvc init
```

### Workflow Complet (Version 1)
```bash
# Preprocessing
PYTHONPATH=. python3 src/data_processing.py \
  --input data/raw/diabetes_v1_original.csv \
  --output data/processed/v1

# Feature Engineering
PYTHONPATH=. python3 src/feature_engineering.py \
  --train data/processed/v1/train.csv \
  --val data/processed/v1/val.csv \
  --test data/processed/v1/test.csv \
  --output data/processed/v1 \
  --dataset diabetes

# Training
PYTHONPATH=. python3 src/train.py \
  --train data/processed/v1/train_engineered.csv \
  --val data/processed/v1/val_engineered.csv \
  --dataset diabetes \
  --output models/v1

# Evaluation
PYTHONPATH=. python3 src/evaluate.py \
  --test data/processed/v1/test_engineered.csv \
  --model models/v1/model.pkl \
  --dataset diabetes

# Optimization (optionnel)
PYTHONPATH=. python3 src/optimize.py \
  --train data/processed/v1/train_engineered.csv \
  --val data/processed/v1/val_engineered.csv \
  --dataset diabetes \
  --output models/v1
```

### Makefile (Simplifié)
```bash
make generate-data     # Générer les datasets
make run-v1           # Pipeline complet V1
make run-v2           # Pipeline complet V2
make run-v3           # Pipeline complet V3
make compare          # Comparer les résultats
make full-workflow    # Tout exécuter
```

---

##  Bonnes Pratiques Implémentées

### 1. Reproductibilité
-  Seeds fixés partout (`random_state=42`)
-  Environnement virtuel avec requirements.txt
-  Configuration centralisée (params.yaml)
-  Versioning complet (Git + DVC)

### 2. Tracking et Monitoring
-  Tous les runs trackés dans MLflow
-  Métriques sauvegardées en JSON
-  Visualisations automatiques
-  Logs structurés

### 3. Modularité
-  Code organisé en modules réutilisables
-  Séparation des responsabilités
-  Configuration externe (params.yaml)
-  Scripts indépendants

### 4. Testing
-  Tests unitaires (pytest)
-  Validation des données
-  CI/CD automatisé
-  Code coverage

### 5. Documentation
-  README complet
-  Docstrings dans le code
-  Guides d'utilisation
-  Résultats documentés

---

##  Références et Ressources

- [Scikit-learn Diabetes Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

---

**Dernière mise à jour** : Décembre 2025  
**Auteur** : Ahmed Ben Abderrazak  
**Version** : 1.0  
**Status** :  Production Ready
