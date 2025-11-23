# Test Suite Documentation

## Overview

This directory contains comprehensive unit tests for the Fraud Detection System. The test suite ensures code quality, reliability, and maintainability.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_model.py            # Model performance and behavior tests
├── test_preprocessing.py    # Data preprocessing and validation tests
└── README.md               # This file
```

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/test_model.py -v
```

### Run with Coverage Report
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Run Single Test Case
```bash
python -m pytest tests/test_model.py::TestFraudDetectionModel::test_model_training -v
```

## Test Categories

### 1. Model Tests (`test_model.py`)

**Purpose:** Validate machine learning model functionality

**Test Cases:**
- `test_model_training` - Ensures model trains without errors
- `test_model_prediction` - Validates prediction outputs
- `test_model_probability` - Checks probability predictions
- `test_model_performance` - Verifies minimum accuracy threshold
- `test_feature_importance` - Validates feature importance calculations
- `test_input_validation` - Tests error handling for invalid inputs
- `test_single_prediction` - Checks single sample predictions
- `test_class_balance_handling` - Tests imbalanced class handling
- `test_reproducibility` - Ensures consistent results with same seed

### 2. Preprocessing Tests (`test_preprocessing.py`)

**Purpose:** Validate data preprocessing and feature engineering

**Test Cases:**
- `test_missing_value_detection` - Identifies missing data
- `test_data_types` - Validates column data types
- `test_categorical_encoding` - Tests one-hot encoding
- `test_amount_scaling` - Validates standardization
- `test_time_feature_engineering` - Tests time-based features
- `test_outlier_detection` - Validates IQR outlier detection
- `test_feature_creation` - Tests derived feature creation
- `test_data_validation` - Checks business rule validation
- `test_aggregation_features` - Tests customer aggregations
- `test_velocity_features` - Validates transaction velocity

## Continuous Integration

Tests run automatically on:
- Every push to `main` or `develop` branches
- Every pull request to `main`
- Multiple Python versions (3.8, 3.9, 3.10)

See `.github/workflows/tests.yml` for CI configuration.

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Template

```python
import unittest

class TestNewFeature(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize test data
        pass
    
    def test_feature_behavior(self):
        """Test specific feature behavior."""
        # Arrange
        expected_result = True
        
        # Act
        actual_result = my_function()
        
        # Assert
        self.assertEqual(actual_result, expected_result)
    
    def tearDown(self):
        """Clean up after tests."""
        pass
```

## Test Coverage Goals

- **Target Coverage:** 80%+
- **Critical Paths:** 100%
- **Error Handling:** 100%

## Best Practices

1. **Isolation:** Each test should be independent
2. **Clarity:** Use descriptive test names and docstrings
3. **Coverage:** Test both success and failure scenarios
4. **Speed:** Keep tests fast for quick feedback
5. **Maintainability:** Update tests when code changes

## Dependencies

```bash
pip install pytest pytest-cov
```

## Troubleshooting

### Import Errors
Ensure the project root is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Test Discovery Issues
Run from project root directory:
```bash
cd /path/to/ds-advanced-fraud-detection
python -m pytest tests/
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain coverage above 80%
4. Document new test cases

## Contact

For questions about tests:
- Check existing test implementations
- Review pytest documentation
- Open an issue on GitHub