"""
Script de génération de données synthétiques pour le dataset Diabetes
Permet de créer différentes versions du dataset avec augmentation de données

EXPLICATION DES COLONNES DU DATASET DIABETES:
============================================

Colonnes Originales (Normalisées entre -2 et 2):
-------------------------------------------------
1. age       : Âge du patient (normalisé)
2. sex       : Sexe du patient (1 = masculin, 2 = féminin, normalisé)
3. bmi       : Body Mass Index - Indice de masse corporelle (poids/taille²)
4. bp        : Blood Pressure - Pression artérielle moyenne
5. s1        : TC - Cholestérol total sérique
6. s2        : LDL - Lipoprotéines de basse densité ("mauvais cholestérol")
7. s3        : HDL - Lipoprotéines de haute densité ("bon cholestérol")
8. s4        : TCH - Ratio cholestérol total / HDL
9. s5        : LTG - Triglycérides (logarithme)
10. s6       : GLU - Glucose sanguin (glycémie)
11. target   : Mesure quantitative de progression du diabète après 1 an (25-346)

Nouvelles Colonnes Calculées (Ajoutées par ce script):
------------------------------------------------------
12. bmi_category       : Catégorie IMC (Underweight/Normal/Overweight/Obese)
13. bp_category        : Catégorie pression (Normal/Elevated/High)
14. cholesterol_ratio  : Ratio LDL/HDL (indicateur risque cardiovasculaire)
15. metabolic_risk     : Score de risque métabolique (combinaison facteurs)
16. age_group          : Groupe d'âge (Young/Middle/Senior)
17. diabetes_risk      : Niveau de risque diabète (Low/Medium/High/Very High)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_diabetes
from src.utils import setup_logging, ensure_dir


logger = setup_logging()


class DiabetesDataGenerator:
    """Générateur de données synthétiques pour Diabetes dataset"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def load_original_dataset(self):
        """Charge le dataset Diabetes original"""
        logger.info("Loading original Diabetes dataset...")
        diabetes = load_diabetes(as_frame=True)
        df = diabetes.frame
        logger.info(f"Original dataset: {len(df)} samples, {len(df.columns)} features")
        return df

    def _add_bmi_category(self, df, df_enriched):
        """Add BMI category feature"""
        bmi_mean, bmi_std = df['bmi'].mean(), df['bmi'].std()
        df_enriched['bmi_category'] = df['bmi'].apply(
            lambda x: 'Underweight' if x < bmi_mean - bmi_std
            else 'Normal' if x < bmi_mean
            else 'Overweight' if x < bmi_mean + bmi_std
            else 'Obese'
        )

    def _add_bp_category(self, df, df_enriched):
        """Add blood pressure category feature"""
        bp_mean, bp_std = df['bp'].mean(), df['bp'].std()
        df_enriched['bp_category'] = df['bp'].apply(
            lambda x: 'Normal' if x < bp_mean
            else 'Elevated' if x < bp_mean + bp_std
            else 'High'
        )

    def _add_metabolic_features(self, df, df_enriched):
        """Add cholesterol ratio and metabolic risk features"""
        df_enriched['cholesterol_ratio'] = df['s2'] / (df['s3'] + 1e-5)

        def normalize(col):
            return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        df_enriched['metabolic_risk'] = (
            0.3 * normalize('bmi') + 0.2 * normalize('bp') + 0.2 * normalize('s2') +
            0.2 * normalize('s5') + 0.1 * normalize('s6')
        )

    def _add_age_group(self, df, df_enriched):
        """Add age group feature"""
        age_mean, age_std = df['age'].mean(), df['age'].std()
        df_enriched['age_group'] = df['age'].apply(
            lambda x: 'Young' if x < age_mean - 0.5 * age_std
            else 'Middle' if x < age_mean + 0.5 * age_std
            else 'Senior'
        )

    def _add_diabetes_risk(self, df, df_enriched):
        """Add diabetes risk level feature"""
        q25, q50, q75 = df['target'].quantile([0.25, 0.50, 0.75])
        df_enriched['diabetes_risk'] = df['target'].apply(
            lambda x: 'Low' if x < q25
            else 'Medium' if x < q50
            else 'High' if x < q75
            else 'Very High'
        )

    def add_derived_features(self, df):
        """
        Ajoute des colonnes dérivées significatives au dataset
        """
        logger.info("Adding derived features to dataset...")
        df_enriched = df.copy()

        self._add_bmi_category(df, df_enriched)
        self._add_bp_category(df, df_enriched)
        self._add_metabolic_features(df, df_enriched)
        self._add_age_group(df, df_enriched)
        self._add_diabetes_risk(df, df_enriched)

        logger.info(f"Added 6 derived features. Total columns: {len(df_enriched.columns)}")
        return df_enriched

    def generate_synthetic_samples(self, df_original, n_samples, variation_level='medium'):
        """
        Génère des échantillons synthétiques basés sur les statistiques du dataset original

        Args:
            df_original: DataFrame original
            n_samples: Nombre d'échantillons à générer
            variation_level: Niveau de variation ('low', 'medium', 'high')

        Returns:
            DataFrame avec échantillons synthétiques
        """
        logger.info(f"Generating {n_samples} synthetic samples with {variation_level} variation...")

        # Définir les facteurs de variation
        variation_factors = {
            'low': 0.05,      # 5% de variation
            'medium': 0.15,   # 15% de variation
            'high': 0.30      # 30% de variation
        }

        noise_factor = variation_factors.get(variation_level, 0.15)

        synthetic_samples = []

        for i in range(n_samples):
            # Sélectionner un échantillon de base aléatoire
            base_sample = df_original.sample(n=1, random_state=self.random_state + i).iloc[0]

            # Créer un nouvel échantillon avec perturbations
            new_sample = {}

            for col in df_original.columns:
                base_value = base_sample[col]

                if col == 'target':
                    # Pour la target, ajouter du bruit gaussien
                    noise = np.random.normal(0, df_original[col].std() * noise_factor)
                    new_value = base_value + noise
                else:
                    # Pour les features, perturbation basée sur la distribution
                    col_std = df_original[col].std()
                    noise = np.random.normal(0, col_std * noise_factor)
                    new_value = base_value + noise

                    # Garder dans les limites raisonnables
                    col_min = df_original[col].min()
                    col_max = df_original[col].max()
                    new_value = np.clip(new_value, col_min * 1.2, col_max * 1.2)

                new_sample[col] = new_value

            synthetic_samples.append(new_sample)

        synthetic_df = pd.DataFrame(synthetic_samples)
        logger.info(f"Generated {len(synthetic_df)} synthetic samples")

        return synthetic_df

    def create_version_1_original(self, output_path):
        """Version 1: Dataset original avec features dérivées (442 samples)"""
        logger.info("Creating Version 1: Original dataset with derived features")

        df = self.load_original_dataset()

        # Ajouter les colonnes dérivées
        df_enriched = self.add_derived_features(df)

        # Afficher statistiques
        self._print_dataset_stats(df_enriched, "Version 1")

        # Sauvegarder
        ensure_dir(Path(output_path).parent)
        df_enriched.to_csv(output_path, index=False)

        logger.info(
            f"Version 1 saved: {len(df_enriched)} samples, {len(df_enriched.columns)} features -> {output_path}")

        return df_enriched

    def create_version_2_augmented(self, output_path, n_synthetic=100):
        """Version 2: Dataset augmenté avec données synthétiques et features dérivées (542 samples)"""
        logger.info(f"Creating Version 2: Augmented dataset (+{n_synthetic} samples)")

        # Charger dataset original
        df_original = self.load_original_dataset()

        # Générer données synthétiques avec variation moyenne
        synthetic_df = self.generate_synthetic_samples(
            df_original,
            n_samples=n_synthetic,
            variation_level='medium'
        )

        # Combiner
        df_augmented = pd.concat([df_original, synthetic_df], ignore_index=True)

        # Mélanger
        df_augmented = df_augmented.sample(
            frac=1, random_state=self.random_state).reset_index(
            drop=True)

        # Ajouter les colonnes dérivées
        df_enriched = self.add_derived_features(df_augmented)

        # Afficher statistiques
        self._print_dataset_stats(df_enriched, "Version 2")

        # Sauvegarder
        ensure_dir(Path(output_path).parent)
        df_enriched.to_csv(output_path, index=False)

        logger.info(
            f"Version 2 saved: {len(df_enriched)} samples, {len(df_enriched.columns)} features -> {output_path}")
        logger.info(f"  - Original: {len(df_original)} samples")
        logger.info(f"  - Synthetic: {len(synthetic_df)} samples")

        return df_enriched

    def create_version_3_varied(self, output_path, n_synthetic=200):
        """Version 3: Dataset avec variations importantes et features dérivées (642 samples)"""
        logger.info(f"Creating Version 3: Dataset with variations (+{n_synthetic} samples)")

        # Charger dataset original
        df_original = self.load_original_dataset()

        # Générer 3 groupes de données synthétiques avec différents niveaux de variation
        synthetic_low = self.generate_synthetic_samples(
            df_original,
            n_samples=n_synthetic // 3,
            variation_level='low'
        )

        synthetic_medium = self.generate_synthetic_samples(
            df_original,
            n_samples=n_synthetic // 3,
            variation_level='medium'
        )

        synthetic_high = self.generate_synthetic_samples(
            df_original,
            n_samples=n_synthetic - (2 * (n_synthetic // 3)),
            variation_level='high'
        )

        # Combiner tous les échantillons
        df_varied = pd.concat([
            df_original,
            synthetic_low,
            synthetic_medium,
            synthetic_high
        ], ignore_index=True)

        # Mélanger
        df_varied = df_varied.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Ajouter les colonnes dérivées
        df_enriched = self.add_derived_features(df_varied)

        # Afficher statistiques
        self._print_dataset_stats(df_enriched, "Version 3")

        # Sauvegarder
        ensure_dir(Path(output_path).parent)
        df_enriched.to_csv(output_path, index=False)

        logger.info(
            f"Version 3 saved: {len(df_enriched)} samples, {len(df_enriched.columns)} features -> {output_path}")
        logger.info(f"  - Original: {len(df_original)} samples")
        logger.info(f"  - Synthetic (low var): {len(synthetic_low)} samples")
        logger.info(f"  - Synthetic (medium var): {len(synthetic_medium)} samples")
        logger.info(f"  - Synthetic (high var): {len(synthetic_high)} samples")

        return df_enriched

    def _print_dataset_stats(self, df, version_name):
        """Affiche les statistiques du dataset"""
        logger.info(f"\n{version_name} Statistics:")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  Total features: {len(df.columns)}")
        logger.info(f"  Target mean: {df['target'].mean():.2f}, std: {df['target'].std():.2f}")

        # Statistiques des colonnes catégorielles
        if 'bmi_category' in df.columns:
            logger.info(f"\n  BMI Distribution:")
            for cat, count in df['bmi_category'].value_counts().items():
                logger.info(f"    {cat}: {count} ({count/len(df)*100:.1f}%)")

        if 'diabetes_risk' in df.columns:
            logger.info(f"\n  Diabetes Risk Distribution:")
            for risk, count in df['diabetes_risk'].value_counts().items():
                logger.info(f"    {risk}: {count} ({count/len(df)*100:.1f}%)")

        if 'metabolic_risk' in df.columns:
            logger.info(
                f"\n  Metabolic Risk Score: mean={df['metabolic_risk'].mean():.3f}, std={df['metabolic_risk'].std():.3f}")

    def generate_all_versions(self, output_dir='data/raw'):
        """Génère toutes les versions du dataset"""
        logger.info("=" * 80)
        logger.info("GENERATING ALL DATASET VERSIONS")
        logger.info("=" * 80)

        ensure_dir(output_dir)

        # Version 1: Original
        v1_path = f"{output_dir}/diabetes_v1_original.csv"
        df_v1 = self.create_version_1_original(v1_path)

        print("\n")

        # Version 2: Augmented (+100)
        v2_path = f"{output_dir}/diabetes_v2_augmented.csv"
        df_v2 = self.create_version_2_augmented(v2_path, n_synthetic=100)

        print("\n")

        # Version 3: Varied (+200)
        v3_path = f"{output_dir}/diabetes_v3_varied.csv"
        df_v3 = self.create_version_3_varied(v3_path, n_synthetic=200)

        # Statistiques comparatives
        logger.info("=" * 80)
        logger.info("SUMMARY OF DATASET VERSIONS")
        logger.info("=" * 80)

        print(f"\nVersion 1 (Original):")
        print(f"  Samples: {len(df_v1)}")
        print(f"  Target mean: {df_v1['target'].mean():.2f}")
        print(f"  Target std: {df_v1['target'].std():.2f}")

        print(f"\nVersion 2 (Augmented +100):")
        print(f"  Samples: {len(df_v2)}")
        print(f"  Target mean: {df_v2['target'].mean():.2f}")
        print(f"  Target std: {df_v2['target'].std():.2f}")

        print(f"\nVersion 3 (Varied +200):")
        print(f"  Samples: {len(df_v3)}")
        print(f"  Target mean: {df_v3['target'].mean():.2f}")
        print(f"  Target std: {df_v3['target'].std():.2f}")

        print("\nAll versions generated successfully!")

        return {
            'v1': df_v1,
            'v2': df_v2,
            'v3': df_v3
        }


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description='Generate Diabetes dataset versions')
    parser.add_argument('--version', type=str, choices=['v1', 'v2', 'v3', 'all'],
                        default='all', help='Dataset version to generate')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')

    args = parser.parse_args()

    # Créer le générateur
    generator = DiabetesDataGenerator(random_state=args.random_state)

    # Générer les versions demandées
    if args.version == 'all':
        generator.generate_all_versions(args.output)
    elif args.version == 'v1':
        generator.create_version_1_original(f"{args.output}/diabetes_v1_original.csv")
    elif args.version == 'v2':
        generator.create_version_2_augmented(f"{args.output}/diabetes_v2_augmented.csv")
    elif args.version == 'v3':
        generator.create_version_3_varied(f"{args.output}/diabetes_v3_varied.csv")


if __name__ == "__main__":
    main()
