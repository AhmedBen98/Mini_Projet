"""
Hyperparameter Optimization with Optuna
Advanced feature: Automated hyperparameter tuning with MLflow integration
"""

import argparse
import pandas as pd
import numpy as np
import time
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.utils import (
    setup_logging, load_params, ensure_dir, save_metrics,
    calculate_regression_metrics, calculate_classification_metrics,
    print_metrics
)


logger = setup_logging()


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna with MLflow tracking"""

    def __init__(self, params: dict, dataset_name: str):
        self.params = params
        self.dataset_name = dataset_name
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.is_classification = None
        self.best_model = None

    def load_data(self, train_file: str, val_file: str):
        """Load training and validation data"""

        logger.info("Loading data...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        # Determine target column
        if 'quality' in train_df.columns:
            target_col = 'quality'
            self.is_classification = True
        elif 'target' in train_df.columns:
            target_col = 'target'
            self.is_classification = False
        elif 'MedHouseVal' in train_df.columns:
            target_col = 'MedHouseVal'
            self.is_classification = False
        else:
            target_col = train_df.columns[-1]
            self.is_classification = False

        # Split features and target
        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]
        self.X_val = val_df.drop(columns=[target_col])
        self.y_val = val_df[target_col]

        logger.info(f"Loaded data - Train: {self.X_train.shape}, Val: {self.X_val.shape}")

    def objective_diabetes(self, trial):
        """Objective function for Diabetes dataset (GradientBoosting)"""

        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }

        # Create model
        model = GradientBoostingRegressor(**params)

        # Cross-validation score (negative MSE)
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=self.params['evaluation']['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # Convert to RMSE for better interpretability
        rmse_scores = np.sqrt(-cv_scores)
        score = -rmse_scores.mean()  # Negative because Optuna maximizes

        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse", rmse_scores.mean())
            mlflow.log_metric("cv_std", rmse_scores.std())

        return score

    def optimize(self, train_file: str, val_file: str, output_dir: str):
        """Run Optuna optimization"""

        # Load data
        self.load_data(train_file, val_file)

        # Setup MLflow
        mlflow.set_tracking_uri(self.params['mlflow']['tracking_uri'])
        experiment_name = f"{self.dataset_name}-optuna-optimization"
        mlflow.set_experiment(experiment_name)

        # Select objective function (Diabetes only)
        if self.dataset_name != 'diabetes':
            raise ValueError(f"Only 'diabetes' dataset is supported, got: {self.dataset_name}")

        objective_func = self.objective_diabetes
        direction = 'maximize'  # Maximizing negative RMSE

        # Start parent MLflow run
        with mlflow.start_run(run_name=f"{self.dataset_name}-optuna-study"):

            logger.info("Starting Optuna optimization...")

            # Create Optuna study
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner
            )

            # Optimize
            n_trials = self.params['optuna']['n_trials']
            timeout = self.params['optuna']['timeout']

            start_time = time.time()
            study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            optimization_time = time.time() - start_time

            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value

            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Best score: {best_score:.6f}")
            logger.info(f"Best parameters: {best_params}")

            # Log optimization results
            mlflow.log_param("n_trials", len(study.trials))
            mlflow.log_param("optimization_time", optimization_time)
            mlflow.log_metric("best_score", best_score)
            mlflow.log_params(best_params)

            # Train final model with best parameters
            logger.info("Training final model with best parameters...")

            self.best_model = GradientBoostingRegressor(**best_params)

            self.best_model.fit(self.X_train, self.y_train)

            # Evaluate on validation set
            y_train_pred = self.best_model.predict(self.X_train)
            y_val_pred = self.best_model.predict(self.X_val)

            if self.is_classification:
                train_metrics = calculate_classification_metrics(
                    self.y_train, y_train_pred,
                    self.best_model.predict_proba(self.X_train)
                )
                val_metrics = calculate_classification_metrics(
                    self.y_val, y_val_pred,
                    self.best_model.predict_proba(self.X_val)
                )
            else:
                train_metrics = calculate_regression_metrics(self.y_train, y_train_pred)
                val_metrics = calculate_regression_metrics(self.y_val, y_val_pred)

            # Log metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)

            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)

            print_metrics(train_metrics, "Training Metrics (Best Model)")
            print_metrics(val_metrics, "Validation Metrics (Best Model)")

            # Save model
            ensure_dir(output_dir)
            model_path = f"{output_dir}/model_optimized.pkl"
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved to {model_path}")

            # Log model to MLflow
            signature = infer_signature(self.X_train, y_train_pred)
            mlflow.sklearn.log_model(
                self.best_model,
                "optimized_model",
                signature=signature,
                registered_model_name=f"{self.dataset_name}_optimized_model"
            )

            # Save optimization results
            optimization_results = {
                'dataset': self.dataset_name,
                'n_trials': len(study.trials),
                'best_score': float(best_score),
                'best_params': best_params,
                'optimization_time': optimization_time,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }

            save_metrics(optimization_results, "metrics/optimization_results.json")
            mlflow.log_artifact("metrics/optimization_results.json")

            # Save Optuna study
            study_path = f"{output_dir}/optuna_study.pkl"
            joblib.dump(study, study_path)
            mlflow.log_artifact(study_path)

            logger.info("Optimization complete!")

            return self.best_model, optimization_results


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Optimize hyperparameters with Optuna')
    parser.add_argument('--train', type=str, default='data/processed/train_engineered.csv')
    parser.add_argument('--val', type=str, default='data/processed/val_engineered.csv')
    parser.add_argument('--dataset', type=str, default='diabetes',
                        choices=['diabetes'])
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--params', type=str, default='params.yaml')

    args = parser.parse_args()

    # Load parameters
    params = load_params(args.params)

    # Optimize
    optimizer = OptunaOptimizer(params, args.dataset)
    optimizer.optimize(args.train, args.val, args.output)


if __name__ == "__main__":
    main()
