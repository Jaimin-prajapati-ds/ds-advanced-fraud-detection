"""
Model training module using ensemble methods and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import (
    RF_N_ESTIMATORS, XGB_N_ESTIMATORS, LGB_N_ESTIMATORS,
    N_SPLITS, RANDOM_STATE, MODELS_DIR
)

class ModelTrainer:
    """Train and manage multiple fraud detection models."""
    
    def __init__(self):
        self.models = {}
        self.cv_scores = {}
        self.best_model = None
        
    def create_random_forest(self):
        """Create Random Forest classifier."""
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    def create_xgboost(self):
        """Create XGBoost classifier."""
        return XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    def create_lightgbm(self):
        """Create LightGBM classifier."""
        return LGBMClassifier(
            n_estimators=LGB_N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    
    def create_stacking_ensemble(self):
        """Create stacking ensemble with multiple base models."""
        base_learners = [
            ('rf', self.create_random_forest()),
            ('xgb', self.create_xgboost()),
            ('lgb', self.create_lightgbm())
        ]
        
        meta_learner = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        )
        
        return StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        )
    
    def train_model(self, X_train, y_train, model_name, model):
        """Train a single model with cross-validation."""
        print(f"\nTraining {model_name}...")
        
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        print(f"{model_name} CV Scores: {cv_scores}")
        print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        model.fit(X_train, y_train)
        self.models[model_name] = model
        self.cv_scores[model_name] = cv_scores
        
        return model, cv_scores
    
    def train_all_models(self, X_train, y_train):
        """Train all models."""
        print(f"\n=== Training ML Models ===")
        
        models_to_train = {
            'Random Forest': self.create_random_forest(),
            'XGBoost': self.create_xgboost(),
            'LightGBM': self.create_lightgbm(),
            'Stacking Ensemble': self.create_stacking_ensemble()
        }
        
        for name, model in models_to_train.items():
            self.train_model(X_train, y_train, name, model)
        
        print(f"\nAll models trained successfully!")
        return self.models
    
    def save_models(self):
        """Save trained models to disk."""
        for name, model in self.models.items():
            filepath = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    print("Model training module ready.")
