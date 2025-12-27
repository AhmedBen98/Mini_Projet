"""
Training Script with MLflow Integration
Trains ML models and logs experiments to MLflow
"""

import argparse
import pandas as pd
import time
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from src.utils import (
    setup_logging, load_params, ensure_dir, save_metrics,
    calculate_regression_metrics, calculate_classification_metrics,
    get_model_info, print_metrics
)


logger = setup_logging()


class ModelTrainer:
    """ML Model training with MLflow tracking"""

    def __init__(self, params: dict):
        self.params = params
        self.model = None
        self.model_type = None

    def get_model(self, dataset_name: str):
        """Get model for Diabetes dataset"""

        if dataset_name != 'diabetes':
            raise ValueError(f"Only 'diabetes' dataset is supported, got: {dataset_name}")

        model_params = self.params['model_diabetes']
        self.model_type = 'regression'
        self.model = GradientBoostingRegressor(
            n_estimators=model_params['n_estimators'],
            learning_rate=model_params['learning_rate'],
            max_depth=model_params['max_depth'],
            min_samples_split=model_params['min_samples_split'],
            min_samples_leaf=model_params['min_samples_leaf'],
            subsample=model_params['subsample'],
            random_state=model_params['random_state']
        )
        logger.info("Using GradientBoostingRegressor for Diabetes")

        return self.model

    def train(self, train_file: str, val_file: str, dataset_name: str,
              output_dir: str):
        """Train model with MLflow tracking"""

        # Load data
        logger.info("Loading training and validation data...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        # Determine target column
        if 'quality' in train_df.columns:
            target_col = 'quality'
        elif 'target' in train_df.columns:
            target_col = 'target'
        elif 'MedHouseVal' in train_df.columns:
            target_col = 'MedHouseVal'
        else:
            target_col = train_df.columns[-1]

        # Split features and target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")

        # Setup MLflow
        mlflow.set_tracking_uri(self.params['mlflow']['tracking_uri'])
        experiment_name = f"{dataset_name}-quality-experiment"
        mlflow.set_experiment(experiment_name)

        # Start MLflow run
        with mlflow.start_run(run_name=f"{dataset_name}-baseline-model"):

            logger.info("Starting MLflow run...")

            # Log parameters
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("n_samples_train", len(X_train))
            mlflow.log_param("n_samples_val", len(X_val))
            mlflow.log_param("n_features", X_train.shape[1])

            # Get and log model
            model = self.get_model(dataset_name)
            mlflow.log_params(model.get_params())

            # Train model
            logger.info("Training model...")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            logger.info(f"Training completed in {training_time:.2f} seconds")
            mlflow.log_metric("training_time_seconds", training_time)

            # Cross-validation on training data
            logger.info("Performing cross-validation...")
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.params['evaluation']['cv_folds'],
                scoring=self.params['evaluation']['scoring']
            )
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            logger.info(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Calculate and log metrics
            if self.model_type == 'classification':
                # Classification metrics
                y_train_pred_proba = model.predict_proba(X_train)
                y_val_pred_proba = model.predict_proba(X_val)

                train_metrics = calculate_classification_metrics(
                    y_train, y_train_pred, y_train_pred_proba
                )
                val_metrics = calculate_classification_metrics(
                    y_val, y_val_pred, y_val_pred_proba
                )

            else:
                # Regression metrics
                train_metrics = calculate_regression_metrics(y_train, y_train_pred)
                val_metrics = calculate_regression_metrics(y_val, y_val_pred)

            # Log training metrics
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)

            # Log validation metrics
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)

            # Print metrics
            print_metrics(train_metrics, "Training Metrics")
            print_metrics(val_metrics, "Validation Metrics")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                logger.info("Top 10 important features:")
                print(feature_importance.head(10))

                # Save and log feature importance
                ensure_dir("plots")
                feature_importance.to_csv("plots/feature_importance.csv", index=False)
                mlflow.log_artifact("plots/feature_importance.csv")

            # Model info
            model_info = get_model_info(model)
            mlflow.log_param("model_type", model_info['model_type'])
            mlflow.log_metric("model_size_mb", model_info['model_size_mb'])

            # Log model with signature
            signature = infer_signature(X_train, y_train_pred)

            if self.model_type == 'classification':
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=f"{dataset_name}_model"
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=f"{dataset_name}_model"
                )

            logger.info("Model logged to MLflow")

            # Save model locally
            ensure_dir(output_dir)
            model_path = f"{output_dir}/model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")

            # Save metrics
            metrics_dict = {
                'dataset': dataset_name,
                'model_type': model_info['model_type'],
                'training_time_seconds': training_time,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }

            save_metrics(metrics_dict, "metrics/train_metrics.json")

            # Log metrics file
            mlflow.log_artifact("metrics/train_metrics.json")

            # Get run ID
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")

            return model, metrics_dict


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--train', type=str, default='data/processed/train_engineered.csv')
    parser.add_argument('--val', type=str, default='data/processed/val_engineered.csv')
    parser.add_argument('--dataset', type=str, default='diabetes',
                        choices=['diabetes'])
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--params', type=str, default='params.yaml')

    args = parser.parse_args()

    # Load parameters
    params = load_params(args.params)

    # Update dataset name in params
    params['dataset']['name'] = args.dataset

    # Train model
    trainer = ModelTrainer(params)
    trainer.train(args.train, args.val, args.dataset, args.output)


if __name__ == "__main__":
    main()
