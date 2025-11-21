"""
Configuration settings for fraud detection project.
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
DATASET_NAME = "creditcard.csv"
RATION_COLS = [f"V{i}" for i in range(1, 29)]  # V1-V28
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"

# Model configuration
TRAIN_TEST_SPLIT = 0.3
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
N_SPLITS = 5  # For K-Fold cross-validation

# SMOTE configuration
SMOTE_K_NEIGHBORS = 5
SMOTE_RANDOM_STATE = 42

# Hyperparameter tuning
N_TRIALS = 50  # For Optuna
N_JOBS = -1  # Use all available cores

# Model hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_CLASS_WEIGHT = "balanced_subsample"

XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

LGB_N_ESTIMATORS = 200
LGB_MAX_DEPTH = 6
LGB_LEARNING_RATE = 0.1
LGB_NUM_LEAVES = 31
LGB_SUBSAMPLE = 0.8
LGB_COLSAMPLE_BYTREE = 0.8

# Evaluation metrics
KEY_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
PROB_THRESHOLD = 0.5

# Logging
LOG_LEVEL = "INFO"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
