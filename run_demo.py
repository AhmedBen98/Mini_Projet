"""
Script de démonstration automatique
Exécute tout le pipeline MLOps pour les 3 versions du dataset
"""

import subprocess
import sys
import time
from pathlib import Path
from src.utils import setup_logging, ensure_dir


logger = setup_logging()


class MLOpsPipelineDemo:
    """Démonstration automatique du pipeline MLOps"""
    
    def __init__(self):
        self.start_time = time.time()
        
    def run_command(self, cmd, description):
        """Exécute une commande shell"""
        logger.info(f"{'='*80}")
        logger.info(f"{description}")
        logger.info(f"Command: {cmd}")
        logger.info(f"{'='*80}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error executing command: {cmd}")
            logger.error(result.stderr)
            return False
        
        logger.info(result.stdout)
        return True
    
    def step_1_generate_datasets(self):
        """Étape 1: Générer les 3 versions du dataset"""
        logger.info("\n" + "="*80)
        logger.info("ÉTAPE 1: GÉNÉRATION DES DATASETS")
        logger.info("="*80 + "\n")
        
        cmd = "python src/generate_diabetes_versions.py --version all"
        return self.run_command(cmd, "Génération des 3 versions du dataset Diabetes")
    
    def step_2_init_git_dvc(self):
        """Étape 2: Initialiser Git et DVC"""
        logger.info("\n" + "="*80)
        logger.info("ÉTAPE 2: INITIALISATION GIT ET DVC")
        logger.info("="*80 + "\n")
        
        commands = [
            ("git init", "Initialisation du repository Git"),
            ("git config user.name 'MLOps Demo'", "Configuration Git user"),
            ("git config user.email 'demo@mlops.com'", "Configuration Git email"),
            ("dvc init", "Initialisation de DVC"),
            ("dvc remote add -d myremote /tmp/dvc-storage", "Ajout du remote DVC"),
        ]
        
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                return False
        
        return True
    
    def step_3_process_version(self, version_num, dataset_file):
        """Étape 3: Traiter une version du dataset"""
        logger.info("\n" + "="*80)
        logger.info(f"ÉTAPE 3.{version_num}: TRAITEMENT VERSION {version_num}")
        logger.info("="*80 + "\n")
        
        v = f"v{version_num}"
        
        # Créer les dossiers
        ensure_dir(f"data/processed/{v}")
        ensure_dir(f"models/{v}")
        ensure_dir(f"metrics/{v}")
        ensure_dir(f"plots/{v}")
        
        commands = [
            (f"python src/data_processing.py --input {dataset_file} --output data/processed/{v}",
             f"Preprocessing - Version {version_num}"),
            
            (f"python src/feature_engineering.py " +
             f"--train data/processed/{v}/train.csv " +
             f"--val data/processed/{v}/val.csv " +
             f"--test data/processed/{v}/test.csv " +
             f"--output data/processed/{v} --dataset diabetes",
             f"Feature Engineering - Version {version_num}"),
            
            (f"python src/train.py " +
             f"--train data/processed/{v}/train_engineered.csv " +
             f"--val data/processed/{v}/val_engineered.csv " +
             f"--dataset diabetes --output models/{v}",
             f"Training - Version {version_num}"),
            
            (f"python src/evaluate.py " +
             f"--test data/processed/{v}/test_engineered.csv " +
             f"--model models/{v}/model.pkl " +
             f"--dataset diabetes",
             f"Evaluation - Version {version_num}"),
        ]
        
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                return False
        
        # Copier les métriques dans le dossier de la version
        ensure_dir(f"metrics/{v}")
        subprocess.run(f"cp metrics/train_metrics.json metrics/{v}/", shell=True)
        subprocess.run(f"cp metrics/test_metrics.json metrics/{v}/", shell=True)
        subprocess.run(f"cp -r plots/* plots/{v}/ 2>/dev/null || true", shell=True)
        
        return True
    
    def step_4_version_control(self, version_num, dataset_file):
        """Étape 4: Versioning Git et DVC"""
        logger.info("\n" + "="*80)
        logger.info(f"ÉTAPE 4.{version_num}: VERSIONING - VERSION {version_num}")
        logger.info("="*80 + "\n")
        
        v = f"v{version_num}"
        
        commands = [
            (f"dvc add {dataset_file}", f"DVC add dataset - Version {version_num}"),
            (f"dvc add models/{v}/model.pkl", f"DVC add model - Version {version_num}"),
            (f"git add {dataset_file}.dvc models/{v}/model.pkl.dvc .gitignore",
             f"Git add DVC files - Version {version_num}"),
            (f"git add metrics/{v}/ plots/{v}/", f"Git add metrics - Version {version_num}"),
            (f"git commit -m 'Version {version_num}: Dataset and model'",
             f"Git commit - Version {version_num}"),
            (f"git tag {v}.0", f"Git tag - Version {version_num}"),
        ]
        
        for cmd, desc in commands:
            self.run_command(cmd, desc)
        
        return True
    
    def step_5_compare_versions(self):
        """Étape 5: Comparer les versions"""
        logger.info("\n" + "="*80)
        logger.info("ÉTAPE 5: COMPARAISON DES VERSIONS")
        logger.info("="*80 + "\n")
        
        cmd = ("python src/compare_versions.py " +
               "--v1-data data/raw/diabetes_v1_original.csv " +
               "--v2-data data/raw/diabetes_v2_augmented.csv " +
               "--v3-data data/raw/diabetes_v3_varied.csv " +
               "--v1-metrics metrics/v1/test_metrics.json " +
               "--v2-metrics metrics/v2/test_metrics.json " +
               "--v3-metrics metrics/v3/test_metrics.json " +
               "--output reports")
        
        return self.run_command(cmd, "Génération du rapport de comparaison")
    
    def run_full_demo(self):
        """Exécute la démo complète"""
        logger.info("\n" + "="*80)
        logger.info("DÉMARRAGE DE LA DÉMONSTRATION MLOPS COMPLÈTE")
        logger.info("Diabetes Dataset - 3 Versions")
        logger.info("="*80 + "\n")
        
        # Étape 1: Générer les datasets
        if not self.step_1_generate_datasets():
            logger.error("Erreur lors de la génération des datasets")
            return False
        
        # Étape 2: Init Git/DVC
        if not self.step_2_init_git_dvc():
            logger.error("Erreur lors de l'initialisation Git/DVC")
            return False
        
        # Traiter les 3 versions
        versions = [
            (1, "data/raw/diabetes_v1_original.csv"),
            (2, "data/raw/diabetes_v2_augmented.csv"),
            (3, "data/raw/diabetes_v3_varied.csv")
        ]
        
        for version_num, dataset_file in versions:
            # Étape 3: Process
            if not self.step_3_process_version(version_num, dataset_file):
                logger.error(f"Erreur lors du traitement de la version {version_num}")
                return False
            
            # Étape 4: Version control
            if not self.step_4_version_control(version_num, dataset_file):
                logger.error(f"Erreur lors du versioning de la version {version_num}")
                return False
        
        # Étape 5: Comparer
        if not self.step_5_compare_versions():
            logger.error("Erreur lors de la comparaison")
            return False
        
        # Résumé final
        elapsed_time = time.time() - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info(" DÉMONSTRATION COMPLÉTÉE AVEC SUCCÈS!")
        logger.info("="*80)
        logger.info(f"Temps total: {elapsed_time:.2f} secondes ({elapsed_time/60:.2f} minutes)")
        logger.info("\n Résultats disponibles dans:")
        logger.info("  - metrics/v1/, metrics/v2/, metrics/v3/")
        logger.info("  - plots/v1/, plots/v2/, plots/v3/")
        logger.info("  - reports/")
        logger.info("\n Pour visualiser les expérimentations MLflow:")
        logger.info("  mlflow ui --port 5000")
        logger.info("\n📖 Voir le rapport de comparaison:")
        logger.info("  cat reports/comparison_report.md")
        logger.info("="*80 + "\n")
        
        return True


def main():
    """Main function"""
    demo = MLOpsPipelineDemo()
    
    try:
        success = demo.run_full_demo()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nDémonstration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
