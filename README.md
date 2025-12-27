# Projet MLOps Avancé - Prédiction de Progression du Diabète

## Vue d'ensemble

Projet MLOps complet utilisant le dataset **Diabetes** (Scikit-learn) pour prédire la progression du diabète basée sur des variables biomédicales. Ce projet démontre l'impact des changements de données sur les performances du modèle à travers **3 versions différentes du dataset**.

## Dataset - 3 Versions Testées

**Diabetes Dataset de Scikit-learn**

### Version 1: Original (Baseline)
- **Taille**: 442 échantillons
- **Source**: Dataset original de scikit-learn
- **Description**: Dataset de référence sans modifications

### Version 2: Augmenté (+100 samples)
- **Taille**: 542 échantillons (442 + 100 synthétiques)
- **Ajout**: 100 échantillons synthétiques générés avec variation moyenne (15%)
- **Objectif**: Tester l'impact de l'augmentation de données

### Version 3: Avec Variations (+200 samples)
- **Taille**: 642 échantillons (442 + 200 synthétiques)
- **Ajout**: 200 échantillons avec 3 niveaux de variation:
  - 66 samples avec faible variation (5%)
  - 66 samples avec variation moyenne (15%)
  - 68 samples avec forte variation (30%)
- **Objectif**: Tester la robustesse du modèle face à la diversité des données

### Caractéristiques Communes
- **Features**: 10 variables biomédicales
  - `age`: Âge
  - `sex`: Sexe
  - `bmi`: Indice de masse corporelle
  - `bp`: Pression artérielle
  - `s1` à `s6`: 6 mesures sériques
- **Target**: Mesure quantitative de la progression du diabète après un an

## Objectif du projet

Construire un pipeline MLOps complet avec:
1. Versioning des données (DVC)
2. Tracking des expérimentations (MLflow)
3. Versioning du code (Git)
4. CI/CD automatisé (GitHub Actions)
5. Fonctionnalités avancées

## Architecture du projet

```
mlops-advanced-project/
├── data/
│   ├── raw/                    # Données brutes
│   ├── processed/              # Données prétraitées
│   └── versions/               # Différentes versions du dataset
├── src/
│   ├── data_processing.py      # Prétraitement des données
│   ├── train.py                # Entraînement du modèle
│   ├── evaluate.py             # Évaluation du modèle
│   └── feature_engineering.py  # Feature engineering
├── models/                     # Modèles entraînés
├── notebooks/                  # Notebooks Jupyter pour exploration
├── tests/                      # Tests unitaires
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml     # CI/CD
├── mlruns/                     # Runs MLflow
├── dvc.yaml                    # Pipeline DVC
├── params.yaml                 # Paramètres du modèle
├── requirements.txt            # Dépendances
└── README.md                   # Documentation
```

## Technologies utilisées

- **Git**: Versioning du code
- **DVC**: Versioning des données et pipelines
- **MLflow**: Tracking des expérimentations
- **GitHub Actions**: CI/CD
- **Python 3.10+**: Langage
- **Scikit-learn**: Machine Learning
- **Pandas/NumPy**: Manipulation de données
- **Matplotlib/Seaborn**: Visualisation

## Fonctionnalités avancées

### 1. Feature Engineering automatisé
- Création de nouvelles features
- Sélection automatique des features importantes
- Transformation polynomiale

### 2. Hyperparameter Tuning avec Optuna
- Optimisation automatique des hyperparamètres
- Intégration avec MLflow
- Tracking de toutes les tentatives

### 3. Model Registry MLflow
- Enregistrement des meilleurs modèles
- Versioning des modèles
- Staging (Development, Staging, Production)

### 4. Data Drift Detection
- Détection de drift dans les données
- Alertes automatiques
- Visualisation des drifts

### 5. Tests automatisés
- Tests unitaires des fonctions
- Tests de validation des données
- Tests de performance du modèle

## Flux de travail

```
1. Acquisition des données → data/raw/
2. Prétraitement → data/processed/
3. Feature Engineering → Nouvelles features
4. Entraînement avec Optuna → Meilleur modèle
5. Évaluation → Métriques MLflow
6. Registry → Promotion du modèle
7. CI/CD → Tests + Déploiement
```

## Quick Start

```bash
# Installation
pip install -r requirements.txt

# Télécharger les données
python src/download_data.py

# Exécuter le pipeline DVC
dvc repro

# Lancer MLflow UI
mlflow ui

# Tester le pipeline localement
pytest tests/
```

## Versions du dataset

Le projet utilisera **3 versions différentes** du dataset :

### Version 1: Dataset complet (Baseline)
- Vin rouge + vin blanc
- Toutes les features
- ~6500 échantillons

### Version 2: Dataset vin rouge uniquement
- Seulement le vin rouge
- ~1600 échantillons
- Comparaison des performances

### Version 3: Dataset avec feature engineering
- Nouvelles features créées
- Features polynomiales
- Sélection des meilleures features

## Résultats attendus

Chaque version produira:
- Métriques de performance (RMSE, MAE, R²)
- Comparaison MLflow
- Analyse de l'impact des changements
- Visualisations

## Auteur

Ahmed Ben Abderrazak

## Licence

MIT
