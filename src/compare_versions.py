"""
Script de comparaison des différentes versions du dataset Diabetes
Compare les performances des modèles sur les 3 versions
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.utils import setup_logging, ensure_dir


logger = setup_logging()


class DatasetComparator:
    """Compare les différentes versions du dataset"""

    def __init__(self):
        self.versions = {}
        self.metrics = {}

    def load_version(self, version_name, filepath):
        """Charge une version du dataset"""
        logger.info(f"Loading {version_name} from {filepath}")
        df = pd.read_csv(filepath)
        self.versions[version_name] = df
        return df

    def load_metrics(self, version_name, metrics_file):
        """Charge les métriques d'une version"""
        if Path(metrics_file).exists():
            with open(metrics_file, 'r') as f:
                self.metrics[version_name] = json.load(f)
            logger.info(f"Loaded metrics for {version_name}")
        else:
            logger.warning(f"Metrics file not found: {metrics_file}")

    def compare_datasets(self, output_dir='reports'):
        """Compare les statistiques des datasets"""
        ensure_dir(output_dir)

        logger.info("Comparing dataset statistics...")

        # Créer rapport de comparaison
        comparison = {}

        for version_name, df in self.versions.items():
            comparison[version_name] = {
                'n_samples': len(df),
                'n_features': len(df.columns) - 1,  # Excluding target
                'target_mean': float(df['target'].mean()),
                'target_std': float(df['target'].std()),
                'target_min': float(df['target'].min()),
                'target_max': float(df['target'].max()),
                'features': {}
            }

            # Statistiques par feature
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col != 'target':
                    comparison[version_name]['features'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }

        # Sauvegarder
        comparison_file = f"{output_dir}/dataset_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Dataset comparison saved to {comparison_file}")

        return comparison

    def plot_target_distributions(self, output_dir='reports'):
        """Plot les distributions de la target pour chaque version"""
        ensure_dir(output_dir)

        logger.info("Plotting target distributions...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (version_name, df) in enumerate(self.versions.items()):
            ax = axes[idx]

            ax.hist(df['target'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(df['target'].mean(), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {df['target'].mean():.2f}")
            ax.set_xlabel('Target Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{version_name}\n({len(df)} samples)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = f"{output_dir}/target_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Target distributions saved to {output_file}")

    def plot_feature_distributions(self, output_dir='reports'):
        """Plot les distributions des features"""
        ensure_dir(output_dir)

        logger.info("Plotting feature distributions...")

        # Sélectionner quelques features clés
        key_features = ['age', 'bmi', 'bp', 's5']

        fig, axes = plt.subplots(len(key_features), 3, figsize=(18, 12))

        for row, feature in enumerate(key_features):
            for col, (version_name, df) in enumerate(self.versions.items()):
                ax = axes[row, col]

                ax.hist(df[feature], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2)
                ax.set_ylabel('Frequency')

                if row == 0:
                    ax.set_title(version_name)
                if col == 0:
                    ax.set_ylabel(f'{feature}\nFrequency')
                if row == len(key_features) - 1:
                    ax.set_xlabel('Value')

                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = f"{output_dir}/feature_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature distributions saved to {output_file}")

    def compare_model_performance(self, output_dir='reports'):
        """Compare les performances des modèles"""
        ensure_dir(output_dir)

        if not self.metrics:
            logger.warning("No metrics loaded for comparison")
            return

        logger.info("Comparing model performance...")

        # Extraire les métriques clés
        metrics_to_compare = ['train_rmse', 'val_rmse', 'train_r2', 'val_r2',
                              'train_mae', 'val_mae']

        comparison_data = []

        for version_name, metrics in self.metrics.items():
            for metric_name in metrics_to_compare:
                if metric_name in metrics:
                    comparison_data.append({
                        'Version': version_name,
                        'Metric': metric_name,
                        'Value': metrics[metric_name]
                    })

        if not comparison_data:
            logger.warning("No metrics data available for comparison")
            return

        df_comparison = pd.DataFrame(comparison_data)

        # Plot comparaison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_compare):
            ax = axes[idx]

            data = df_comparison[df_comparison['Metric'] == metric]

            if not data.empty:
                versions = data['Version'].tolist()
                values = data['Value'].tolist()

                bars = ax.bar(versions, values, color=['#3498db', '#2ecc71', '#e74c3c'])
                ax.set_ylabel('Value')
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3, axis='y')

                # Ajouter les valeurs sur les barres
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom')

        plt.tight_layout()
        output_file = f"{output_dir}/model_performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Model performance comparison saved to {output_file}")

    def generate_report(self, output_dir='reports'):
        """Génère un rapport complet de comparaison"""
        ensure_dir(output_dir)

        logger.info("Generating comparison report...")

        # Comparer datasets
        dataset_stats = self.compare_datasets(output_dir)

        # Générer plots
        self.plot_target_distributions(output_dir)
        self.plot_feature_distributions(output_dir)

        if self.metrics:
            self.compare_model_performance(output_dir)

        # Créer rapport Markdown
        report_lines = [
            "# Rapport de Comparaison - Versions du Dataset Diabetes",
            "",
            "## Statistiques des Datasets",
            ""
        ]

        # Tableau comparatif
        report_lines.append("| Métrique | Version 1 | Version 2 | Version 3 |")
        report_lines.append("|----------|-----------|-----------|-----------|")

        metrics_names = [
            ('n_samples', 'Nombre d\'échantillons'),
            ('target_mean', 'Target - Moyenne'),
            ('target_std', 'Target - Écart-type'),
            ('target_min', 'Target - Min'),
            ('target_max', 'Target - Max')
        ]

        for key, label in metrics_names:
            values = [f"{dataset_stats[v][key]:.2f}" if isinstance(dataset_stats[v][key], float)
                      else str(dataset_stats[v][key])
                      for v in ['Version 1', 'Version 2', 'Version 3']]
            report_lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")

        report_lines.append("")
        report_lines.append("## Visualisations")
        report_lines.append("")
        report_lines.append("### Distributions de la Target")
        report_lines.append("![Target Distributions](target_distributions.png)")
        report_lines.append("")
        report_lines.append("### Distributions des Features")
        report_lines.append("![Feature Distributions](feature_distributions.png)")

        if self.metrics:
            report_lines.append("")
            report_lines.append("## Performance des Modèles")
            report_lines.append("")
            report_lines.append("![Model Performance](model_performance_comparison.png)")
            report_lines.append("")

            # Ajouter tableau de métriques
            report_lines.append("### Métriques Détaillées")
            report_lines.append("")
            report_lines.append("| Métrique | Version 1 | Version 2 | Version 3 |")
            report_lines.append("|----------|-----------|-----------|-----------|")

            for metric in ['val_rmse', 'val_r2', 'val_mae']:
                values = []
                for v in ['Version 1', 'Version 2', 'Version 3']:
                    if v in self.metrics and metric in self.metrics[v]:
                        values.append(f"{self.metrics[v][metric]:.4f}")
                    else:
                        values.append("N/A")

                report_lines.append(f"| {metric} | {values[0]} | {values[1]} | {values[2]} |")

        # Sauvegarder le rapport
        report_file = f"{output_dir}/comparison_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Comparison report saved to {report_file}")


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Compare Diabetes dataset versions')
    parser.add_argument('--v1-data', type=str,
                        default='data/raw/diabetes_v1_original.csv')
    parser.add_argument('--v2-data', type=str,
                        default='data/raw/diabetes_v2_augmented.csv')
    parser.add_argument('--v3-data', type=str,
                        default='data/raw/diabetes_v3_varied.csv')
    parser.add_argument('--v1-metrics', type=str,
                        default='metrics/v1/train_metrics.json')
    parser.add_argument('--v2-metrics', type=str,
                        default='metrics/v2/train_metrics.json')
    parser.add_argument('--v3-metrics', type=str,
                        default='metrics/v3/train_metrics.json')
    parser.add_argument('--output', type=str, default='reports')

    args = parser.parse_args()

    # Créer le comparateur
    comparator = DatasetComparator()

    # Charger les datasets
    comparator.load_version('Version 1', args.v1_data)
    comparator.load_version('Version 2', args.v2_data)
    comparator.load_version('Version 3', args.v3_data)

    # Charger les métriques si disponibles
    comparator.load_metrics('Version 1', args.v1_metrics)
    comparator.load_metrics('Version 2', args.v2_metrics)
    comparator.load_metrics('Version 3', args.v3_metrics)

    # Générer le rapport
    comparator.generate_report(args.output)

    print("\n" + "=" * 80)
    print("Comparison report generated successfully!")
    print(f"Check the '{args.output}' directory for results")
    print("=" * 80)


if __name__ == "__main__":
    main()
