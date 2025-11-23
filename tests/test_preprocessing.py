"""Unit tests for data preprocessing functions.

Tests cover:
- Data cleaning and validation
- Feature engineering
- Handling missing values
- Scaling and normalization
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TestDataPreprocessing(unittest.TestCase):
    """Test suite for data preprocessing operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample transaction data
        self.sample_data = pd.DataFrame({
            'transaction_amount': [100.0, 250.0, 50.0, 1000.0, 75.0],
            'transaction_hour': [14, 23, 9, 2, 18],
            'merchant_category': ['retail', 'online', 'retail', 'online', 'gas'],
            'customer_age': [35, 42, 28, None, 51],
            'account_age_days': [365, 730, 180, 1095, 450]
        })
    
    def test_missing_value_detection(self):
        """Test detection of missing values."""
        missing_count = self.sample_data.isnull().sum()
        self.assertEqual(missing_count['customer_age'], 1)
        self.assertEqual(missing_count['transaction_amount'], 0)
    
    def test_data_types(self):
        """Test that data types are correct."""
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['transaction_amount']))
        self.assertTrue(pd.api.types.is_object_dtype(self.sample_data['merchant_category']))
    
    def test_categorical_encoding(self):
        """Test categorical variable encoding."""
        # One-hot encode merchant_category
        encoded = pd.get_dummies(self.sample_data['merchant_category'], prefix='merchant')
        
        # Check number of columns created
        unique_categories = self.sample_data['merchant_category'].nunique()
        self.assertEqual(encoded.shape[1], unique_categories)
        
        # Check all values are 0 or 1
        self.assertTrue(encoded.isin([0, 1]).all().all())
    
    def test_amount_scaling(self):
        """Test transaction amount standardization."""
        scaler = StandardScaler()
        amounts = self.sample_data[['transaction_amount']]
        scaled_amounts = scaler.fit_transform(amounts)
        
        # Check scaled data has mean ≈ 0 and std ≈ 1
        self.assertAlmostEqual(scaled_amounts.mean(), 0.0, places=10)
        self.assertAlmostEqual(scaled_amounts.std(), 1.0, places=10)
    
    def test_time_feature_engineering(self):
        """Test time-based feature creation."""
        # Create is_night flag (hour < 6 or hour >= 22)
        is_night = ((self.sample_data['transaction_hour'] < 6) | 
                   (self.sample_data['transaction_hour'] >= 22)).astype(int)
        
        # Check night transactions are correctly identified
        self.assertEqual(is_night.iloc[1], 1)  # 23:00 is night
        self.assertEqual(is_night.iloc[2], 0)  # 9:00 is day
        self.assertEqual(is_night.iloc[3], 1)  # 2:00 is night
    
    def test_outlier_detection(self):
        """Test outlier detection using IQR method."""
        amounts = self.sample_data['transaction_amount']
        Q1 = amounts.quantile(0.25)
        Q3 = amounts.quantile(0.75)
        IQR = Q3 - Q1
        
        # Identify outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (amounts < lower_bound) | (amounts > upper_bound)
        
        # Check outlier detection works
        self.assertIsInstance(outliers, pd.Series)
        self.assertEqual(len(outliers), len(amounts))
    
    def test_feature_creation(self):
        """Test derived feature creation."""
        # Create amount per account day feature
        self.sample_data['amount_per_account_day'] = (
            self.sample_data['transaction_amount'] / 
            self.sample_data['account_age_days']
        )
        
        # Check new feature exists
        self.assertIn('amount_per_account_day', self.sample_data.columns)
        
        # Check calculation is correct for first row
        expected_value = 100.0 / 365
        self.assertAlmostEqual(
            self.sample_data['amount_per_account_day'].iloc[0],
            expected_value,
            places=6
        )
    
    def test_data_validation(self):
        """Test data validation rules."""
        # Transaction amounts should be positive
        self.assertTrue((self.sample_data['transaction_amount'] > 0).all())
        
        # Transaction hour should be between 0 and 23
        self.assertTrue(
            (self.sample_data['transaction_hour'] >= 0).all() and
            (self.sample_data['transaction_hour'] <= 23).all()
        )
        
        # Account age should be non-negative
        self.assertTrue((self.sample_data['account_age_days'] >= 0).all())


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for advanced feature engineering."""
    
    def setUp(self):
        """Set up test data."""
        self.transactions = pd.DataFrame({
            'customer_id': [1, 1, 1, 2, 2],
            'amount': [100, 150, 200, 50, 75],
            'timestamp': pd.date_range('2025-01-01', periods=5, freq='D')
        })
    
    def test_aggregation_features(self):
        """Test customer-level aggregation features."""
        # Calculate average transaction amount per customer
        customer_avg = self.transactions.groupby('customer_id')['amount'].mean()
        
        # Check aggregation works correctly
        self.assertAlmostEqual(customer_avg.loc[1], 150.0)  # (100+150+200)/3
        self.assertAlmostEqual(customer_avg.loc[2], 62.5)   # (50+75)/2
    
    def test_time_since_features(self):
        """Test time-since-last-transaction feature."""
        # Sort by customer and timestamp
        df = self.transactions.sort_values(['customer_id', 'timestamp'])
        
        # Calculate days since last transaction
        df['days_since_last'] = df.groupby('customer_id')['timestamp'].diff().dt.days
        
        # First transaction for each customer should have NaN
        customer_firsts = df.groupby('customer_id').first()
        self.assertTrue(pd.isna(customer_firsts['days_since_last']).all())
    
    def test_velocity_features(self):
        """Test transaction velocity calculation."""
        # Count transactions per customer
        transaction_count = self.transactions.groupby('customer_id').size()
        
        # Check counts are correct
        self.assertEqual(transaction_count.loc[1], 3)
        self.assertEqual(transaction_count.loc[2], 2)


if __name__ == '__main__':
    unittest.main()