"""
Model evaluation module with comprehensive metrics and visualization.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from config import REPORTS_DIR

class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
        }
        return metrics
    
    def get_confusion_matrix(self, y_true, y_pred):
        """Generate confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            'TP': tp, 'FP': fp,
            'FN': fn, 'TN': tn,
            'Sensitivity (Recall)': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0
        }
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model comprehensively."""
        print(f"\nEvaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        cm_info = self.get_confusion_matrix(y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "ROC-AUC: N/A")
        
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm_info,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def compare_models(self):
        """Compare all evaluated models."""
        print(f"\n=== Model Comparison ===")
        
        comparison_df = pd.DataFrame({
            model: result['metrics'] 
            for model, result in self.results.items()
        }).T
        
        print(f"\n{comparison_df}")
        
        best_model = comparison_df['f1'].idxmax()
        print(f"\nBest Model (by F1-Score): {best_model}")
        
        return comparison_df
    
    def save_results(self):
        """Save evaluation results to files."""
        results_file = REPORTS_DIR / "model_evaluation_results.json"
        
        results_to_save = {}
        for model, result in self.results.items():
            results_to_save[model] = {
                'metrics': {k: float(v) if v is not None else None 
                           for k, v in result['metrics'].items()},
                'confusion_matrix': {k: int(v) if isinstance(v, (int, np.integer)) 
                                    else float(v) 
                                    for k, v in result['confusion_matrix'].items()}
            }
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print("Model evaluation module ready.")
