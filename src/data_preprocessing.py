"""
Data preprocessing module for fraud detection project.
Handles data loading, cleaning, and basic transformations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_NAME,
    TRAIN_TEST_SPLIT, RANDOM_STATE, TARGET_COL, TIME_COL, AMOUNT_COL
)

class DataPreprocessor:
    """Handle data preprocessing tasks."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.feature_names = None
        
    def load_data(self, filepath=None):
        """Load credit card fraud dataset."""
        if filepath is None:
            filepath = RAW_DATA_DIR / DATASET_NAME
        
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Class distribution:\n{df[TARGET_COL].value_counts()}")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values if any."""
        print(f"\nHandling missing values...")
        print(f"Missing values before: {df.isnull().sum().sum()}")
        df = df.dropna()
        print(f"Missing values after: {df.isnull().sum().sum()}")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate records."""
        print(f"\nRemoving duplicates...")
        print(f"Duplicates before: {df.duplicated().sum()}")
        df = df.drop_duplicates()
        print(f"Duplicates after: {df.duplicated().sum()}")
        return df
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features using StandardScaler."""
        print(f"\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def split_data(self, df, test_size=TRAIN_TEST_SPLIT):
        """Split data into train and test sets."""
        print(f"\nSplitting data with test_size={test_size}...")
        
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Train set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Train fraud rate: {y_train.mean():.4f}")
        print(f"Test fraud rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess(self, filepath=None):
        """Run complete preprocessing pipeline."""
        df = self.load_data(filepath)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"\nPreprocessing completed!")
        print(f"Processed data shape: ({X_train_scaled.shape}, {X_test_scaled.shape})")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': X_train.columns.tolist(),
            'scaler': self.scaler
        }

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess()
    print(f"\nData preprocessing successful!")
    print(f"Ready for model training.")
