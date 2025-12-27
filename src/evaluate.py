"""
Evaluation Script
Evaluates trained model on test set
"""

import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.utils import (
    setup_logging, load_params, ensure_dir, save_metrics,
    calculate_regression_metrics, calculate_classification_metrics,
    print_metrics
)


logger = setup_logging()


class ModelEvaluator:
    """Model evaluation with visualization"""

    def __init__(self, params: dict):
        self.params = params

    def load_model(self, model_path: str):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model

    def plot_confusion_matrix(self, y_true, y_pred, output_path: str):
        """Plot and save confusion matrix for classification"""

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved to {output_path}")

    def plot_feature_importance(self, model, feature_names, output_path: str):
        """Plot feature importance"""

        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance plot saved to {output_path}")

    def plot_regression_results(self, y_true, y_pred, output_path: str):
        """Plot regression predictions vs actual"""

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Scatter plot: Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True, alpha=0.3)

        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Regression plots saved to {output_path}")

    def plot_prediction_distribution(self, y_true, y_pred, output_path: str):
        """Plot distribution of predictions vs actual values"""

        plt.figure(figsize=(10, 6))

        plt.hist(y_true, bins=30, alpha=0.5, label='Actual', color='blue')
        plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='red')

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution: Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Distribution plot saved to {output_path}")

    def evaluate(self, test_file: str, model_path: str, dataset_name: str, output_dir: str = None):
        """Evaluate model on test set"""

        # Load test data
        logger.info(f"Loading test data from {test_file}")
        test_df = pd.read_csv(test_file)

        # Set output directories
        if output_dir:
            plots_dir = f"plots/{output_dir}"
            metrics_dir = f"metrics/{output_dir}"
        else:
            plots_dir = "plots"
            metrics_dir = "metrics"

        ensure_dir(plots_dir)
        ensure_dir(metrics_dir)

        # Determine target column
        if 'quality' in test_df.columns:
            target_col = 'quality'
            is_classification = True
        elif 'target' in test_df.columns:
            target_col = 'target'
            is_classification = False
        elif 'MedHouseVal' in test_df.columns:
            target_col = 'MedHouseVal'
            is_classification = False
        else:
            target_col = test_df.columns[-1]
            is_classification = False

        # Split features and target
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        logger.info(f"Test data shape: {X_test.shape}")

        # Load model
        model = self.load_model(model_path)

        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)

        # Calculate metrics
        if is_classification:
            y_pred_proba = model.predict_proba(X_test)
            test_metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        else:
            test_metrics = calculate_regression_metrics(y_test, y_pred)

        # Print metrics
        print_metrics(test_metrics, "Test Set Metrics")

        # Generate plots
        if self.params['evaluation']['generate_plots']:

            if is_classification:
                # Confusion matrix
                self.plot_confusion_matrix(
                    y_test, y_pred, f"{plots_dir}/confusion_matrix.png"
                )
            else:
                # Regression plots
                self.plot_regression_results(
                    y_test.values, y_pred, f"{plots_dir}/regression_results.png"
                )

            # Feature importance
            self.plot_feature_importance(
                model, X_test.columns, f"{plots_dir}/feature_importance.png"
            )

            # Prediction distribution
            self.plot_prediction_distribution(
                y_test.values, y_pred, f"{plots_dir}/prediction_distribution.png"
            )

        # Save predictions
        if self.params['evaluation']['save_predictions']:
            predictions_df = pd.DataFrame({
                'actual': y_test.values,
                'predicted': y_pred
            })
            predictions_df.to_csv(f"{metrics_dir}/predictions.csv", index=False)
            logger.info(f"Predictions saved to {metrics_dir}/predictions.csv")

        # Save metrics
        metrics_dict = {
            'dataset': dataset_name,
            'n_samples_test': len(X_test),
            **test_metrics
        }
        save_metrics(metrics_dict, f"{metrics_dir}/test_metrics.json")

        logger.info("Evaluation complete!")

        return test_metrics


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Evaluate ML model')
    parser.add_argument('--test', type=str, default='data/processed/test_engineered.csv')
    parser.add_argument('--model', type=str, default='models/model.pkl')
    parser.add_argument('--dataset', type=str, default='diabetes',
                        choices=['diabetes'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots and metrics (e.g., v1, v2, v3)')
    parser.add_argument('--params', type=str, default='params.yaml')

    args = parser.parse_args()

    # Load parameters
    params = load_params(args.params)

    # Evaluate model
    evaluator = ModelEvaluator(params)
    evaluator.evaluate(args.test, args.model, args.dataset, args.output)


if __name__ == "__main__":
    main()
