"""
Feature Engineering Script
Creates new features and performs feature selection
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from src.utils import setup_logging, load_params, ensure_dir


logger = setup_logging()


class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self, params: dict):
        self.params = params
        self.poly_features = None
        self.feature_selector = None
        self.selected_features = None
        
    def create_polynomial_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                                   X_test: pd.DataFrame) -> tuple:
        """Create polynomial features"""
        
        if not self.params['feature_engineering']['polynomial_features']:
            logger.info("Polynomial features disabled")
            return X_train, X_val, X_test
        
        degree = self.params['feature_engineering']['degree']
        interaction_only = self.params['feature_engineering']['interaction_only']
        
        logger.info(f"Creating polynomial features (degree={degree}, interaction_only={interaction_only})")
        
        self.poly_features = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        
        # Fit on training data
        X_train_poly = self.poly_features.fit_transform(X_train)
        X_val_poly = self.poly_features.transform(X_val)
        X_test_poly = self.poly_features.transform(X_test)
        
        # Get feature names
        feature_names = self.poly_features.get_feature_names_out(X_train.columns)
        
        # Convert back to DataFrame
        X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)
        X_val_poly = pd.DataFrame(X_val_poly, columns=feature_names, index=X_val.index)
        X_test_poly = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)
        
        logger.info(f"Polynomial features created: {X_train_poly.shape[1]} features")
        
        return X_train_poly, X_val_poly, X_test_poly
    
    def create_domain_features(self, X: pd.DataFrame, dataset_type: str = 'diabetes') -> pd.DataFrame:
        """Create domain-specific features for Diabetes dataset"""
        
        X_new = X.copy()
        
        # Diabetes-specific features
        if 'bmi' in X.columns and 'age' in X.columns:
            X_new['bmi_age_interaction'] = X['bmi'] * X['age']
        
        if 'bp' in X.columns and 'bmi' in X.columns:
            X_new['bp_bmi_ratio'] = X['bp'] / (X['bmi'] + 1e-5)
        
        logger.info(f"Created {len(X_new.columns) - len(X.columns)} diabetes-specific features")
        
        return X_new
    
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, X_test: pd.DataFrame,
                       is_classification: bool = True) -> tuple:
        """Select best features using statistical tests"""
        
        if not self.params['feature_engineering']['feature_selection']:
            logger.info("Feature selection disabled")
            return X_train, X_val, X_test
        
        n_features = min(
            self.params['feature_engineering']['n_features_to_select'],
            X_train.shape[1]
        )
        
        logger.info(f"Selecting top {n_features} features")
        
        # Choose scoring function
        score_func = f_classif if is_classification else f_regression
        
        self.feature_selector = SelectKBest(score_func=score_func, k=n_features)
        
        # Fit on training data
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[selected_mask].tolist()
        
        logger.info(f"Selected features: {self.selected_features}")
        
        # Convert back to DataFrame
        X_train_selected = pd.DataFrame(
            X_train_selected, columns=self.selected_features, index=X_train.index
        )
        X_val_selected = pd.DataFrame(
            X_val_selected, columns=self.selected_features, index=X_val.index
        )
        X_test_selected = pd.DataFrame(
            X_test_selected, columns=self.selected_features, index=X_test.index
        )
        
        return X_train_selected, X_val_selected, X_test_selected
    
    def engineer(self, train_file: str, val_file: str, test_file: str,
                output_dir: str, dataset_type: str = 'diabetes'):
        """Main feature engineering pipeline"""
        
        # Load processed data
        logger.info("Loading processed data...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        # Determine target column
        if 'quality' in train_df.columns:
            target_col = 'quality'
            is_classification = True
        elif 'target' in train_df.columns:
            target_col = 'target'
            is_classification = False
        elif 'MedHouseVal' in train_df.columns:
            target_col = 'MedHouseVal'
            is_classification = False
        else:
            target_col = train_df.columns[-1]
            is_classification = False
        
        # Split features and target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Create domain-specific features
        X_train = self.create_domain_features(X_train, dataset_type)
        X_val = self.create_domain_features(X_val, dataset_type)
        X_test = self.create_domain_features(X_test, dataset_type)
        
        # Create polynomial features (if enabled)
        X_train, X_val, X_test = self.create_polynomial_features(X_train, X_val, X_test)
        
        # Feature selection (if enabled)
        X_train, X_val, X_test = self.select_features(
            X_train, y_train, X_val, X_test, is_classification
        )
        
        # Save engineered data
        ensure_dir(output_dir)
        
        train_engineered = pd.concat([X_train, y_train], axis=1)
        val_engineered = pd.concat([X_val, y_val], axis=1)
        test_engineered = pd.concat([X_test, y_test], axis=1)
        
        train_engineered.to_csv(f"{output_dir}/train_engineered.csv", index=False)
        val_engineered.to_csv(f"{output_dir}/val_engineered.csv", index=False)
        test_engineered.to_csv(f"{output_dir}/test_engineered.csv", index=False)
        
        logger.info(f"Feature engineering complete. Final shape: {X_train.shape}")
        logger.info(f"Engineered data saved to {output_dir}")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Feature engineering')
    parser.add_argument('--train', type=str, default='data/processed/train.csv')
    parser.add_argument('--val', type=str, default='data/processed/val.csv')
    parser.add_argument('--test', type=str, default='data/processed/test.csv')
    parser.add_argument('--output', type=str, default='data/processed')
    parser.add_argument('--dataset', type=str, default='diabetes',
                       choices=['diabetes'])
    parser.add_argument('--params', type=str, default='params.yaml')
    
    args = parser.parse_args()
    
    # Load parameters
    params = load_params(args.params)
    
    # Engineer features
    engineer = FeatureEngineer(params)
    engineer.engineer(args.train, args.val, args.test, args.output, args.dataset)


if __name__ == "__main__":
    main()
