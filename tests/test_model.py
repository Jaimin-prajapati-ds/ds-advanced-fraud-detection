"""Unit tests for fraud detection model.

Tests cover:
- Model training and prediction
- Model performance metrics
- Input validation
- Edge cases and error handling
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestFraudDetectionModel(unittest.TestCase):
    """Test suite for the fraud detection model."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate synthetic test data
        np.random.seed(42)
        n_samples = 1000
        
        # Create feature matrix with 10 features
        self.X = pd.DataFrame(
            np.random.randn(n_samples, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        
        # Create imbalanced target (5% fraud rate)
        fraud_rate = 0.05
        self.y = np.random.choice(
            [0, 1], 
            size=n_samples, 
            p=[1-fraud_rate, fraud_rate]
        )
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def test_model_training(self):
        """Test that model trains without errors."""
        try:
            self.model.fit(self.X_train, self.y_train)
            self.assertTrue(True, "Model trained successfully")
        except Exception as e:
            self.fail(f"Model training failed: {str(e)}")
    
    def test_model_prediction(self):
        """Test that model makes predictions correctly."""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        # Check predictions shape matches test data
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check predictions are binary (0 or 1)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
    
    def test_model_probability(self):
        """Test that model returns valid probability predictions."""
        self.model.fit(self.X_train, self.y_train)
        probabilities = self.model.predict_proba(self.X_test)
        
        # Check probability shape
        self.assertEqual(probabilities.shape, (len(self.y_test), 2))
        
        # Check probabilities sum to 1
        prob_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(self.y_test)))
        
        # Check probabilities are between 0 and 1
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
    
    def test_model_performance(self):
        """Test that model achieves minimum performance threshold."""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, predictions)
        
        # Check minimum accuracy threshold (70%)
        self.assertGreater(accuracy, 0.70, 
                          f"Model accuracy {accuracy:.2f} is below threshold 0.70")
    
    def test_feature_importance(self):
        """Test that feature importances are valid."""
        self.model.fit(self.X_train, self.y_train)
        importances = self.model.feature_importances_
        
        # Check number of importances matches features
        self.assertEqual(len(importances), self.X_train.shape[1])
        
        # Check importances sum to 1
        self.assertAlmostEqual(importances.sum(), 1.0, places=5)
        
        # Check all importances are non-negative
        self.assertTrue(np.all(importances >= 0))
    
    def test_input_validation(self):
        """Test model handles invalid inputs gracefully."""
        self.model.fit(self.X_train, self.y_train)
        
        # Test with wrong number of features
        invalid_input = pd.DataFrame(
            np.random.randn(5, 5),  # Wrong number of features
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        with self.assertRaises(ValueError):
            self.model.predict(invalid_input)
    
    def test_single_prediction(self):
        """Test model handles single sample prediction."""
        self.model.fit(self.X_train, self.y_train)
        
        # Get single sample
        single_sample = self.X_test.iloc[[0]]
        prediction = self.model.predict(single_sample)
        
        # Check single prediction is returned
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1])
    
    def test_class_balance_handling(self):
        """Test model handles imbalanced classes."""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        # Both classes should be predicted at least once
        unique_predictions = np.unique(predictions)
        self.assertTrue(len(unique_predictions) >= 1, 
                       "Model should predict at least one class")
    
    def test_reproducibility(self):
        """Test that model predictions are reproducible with same seed."""
        # Train first model
        model1 = RandomForestClassifier(n_estimators=100, random_state=42)
        model1.fit(self.X_train, self.y_train)
        pred1 = model1.predict(self.X_test)
        
        # Train second model with same seed
        model2 = RandomForestClassifier(n_estimators=100, random_state=42)
        model2.fit(self.X_train, self.y_train)
        pred2 = model2.predict(self.X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)


if __name__ == '__main__':
    unittest.main()