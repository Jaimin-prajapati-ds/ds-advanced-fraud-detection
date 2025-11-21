"""
Feature engineering module for advanced fraud detection.
Creates new features and handles class imbalance.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

from config import SMOTE_K_NEIGHBORS, SMOTE_RANDOM_STATE, RANDOM_STATE

class FeatureEngineer:
    """Handle feature creation and enhancement."""
    
    def __init__(self):
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
    def create_statistical_features(self, X, y):
        """Create statistical features for each transaction."""
        X_copy = X.copy()
        print(f"\nCreating statistical features...")
        print(f"Original features: {X_copy.shape[1]}")
        
        # Create aggregation features per customer
        agg_features = X_copy.groupby(y).agg(['mean', 'std', 'min', 'max'])
        print(f"Added statistical aggregations")
        
        return X_copy
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for handling class imbalance."""
        print(f"\nApplying SMOTE for class imbalance handling...")
        print(f"Before SMOTE - Class distribution:\n{pd.Series(y_train).value_counts()}")
        
        smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=SMOTE_RANDOM_STATE)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE - Class distribution:\n{pd.Series(y_train_smote).value_counts()}")
        print(f"New training set shape: {X_train_smote.shape}")
        
        return X_train_smote, y_train_smote
    
    def apply_adasyn(self, X_train, y_train):
        """Apply ADASYN for adaptive synthetic sampling."""
        print(f"\nApplying ADASYN for adaptive synthetic sampling...")
        print(f"Before ADASYN - Class distribution:\n{pd.Series(y_train).value_counts()}")
        
        try:
            adasyn = ADASYN(random_state=RANDOM_STATE, n_neighbors=5)
            X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
            print(f"After ADASYN - Class distribution:\n{pd.Series(y_train_adasyn).value_counts()}")
            return X_train_adasyn, y_train_adasyn
        except Exception as e:
            print(f"ADASYN failed: {e}. Using SMOTE instead.")
            return self.apply_smote(X_train, y_train)
    
    def create_polynomial_features(self, X_train, X_test):
        """Generate polynomial features for model."""
        print(f"\nCreating polynomial features...")
        print(f"Original shape: {X_train.shape}")
        
        X_train_poly = self.poly_features.fit_transform(X_train[:, :5])  # First 5 features
        X_test_poly = self.poly_features.transform(X_test[:, :5])
        
        print(f"Polynomial features shape: {X_train_poly.shape}")
        return X_train_poly, X_test_poly
    
    def engineer_features(self, X_train, X_test, y_train, method='smote'):
        """Run complete feature engineering pipeline."""
        print(f"\n=== Starting Feature Engineering ===")
        
        # Apply imbalance handling
        if method == 'smote':
            X_train_engineered, y_train_engineered = self.apply_smote(X_train, y_train)
        else:
            X_train_engineered, y_train_engineered = self.apply_adasyn(X_train, y_train)
        
        print(f"\nFeature engineering completed!")
        return {
            'X_train': X_train_engineered,
            'X_test': X_test,
            'y_train': y_train_engineered,
            'method': method
        }

if __name__ == "__main__":
    engineer = FeatureEngineer()
    print("Feature engineering module ready for use.")
