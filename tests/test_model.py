"""
Unit Tests for Credit Risk Model - Task 5
Tests for model training and prediction functions

Author: Bati Bank Analytics Team
Date: June 2, 2026
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import joblib

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================
# SIMPLE EVALUATION FUNCTION FOR TESTING
# ============================================

def simple_evaluate(y_true, y_pred, y_pred_proba):
    """Simple evaluation function for testing"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing"""
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    y_pred_proba = np.array([0.9, 0.2, 0.85, 0.3, 0.95, 0.4, 0.1, 0.2, 0.88, 0.15])
    return y_true, y_pred, y_pred_proba


@pytest.fixture
def sample_model_data():
    """Create sample data for model testing"""
    return pd.DataFrame({
        'feature1': [100, 200, 300, 400, 500],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': [1, 2, 3, 4, 5],
        'is_high_risk': [1, 0, 1, 0, 1]
    })


# ============================================
# TEST 1: EVALUATION FUNCTION
# ============================================

class TestEvaluationFunction:
    """Tests for model evaluation function"""
    
    def test_returns_dictionary(self, sample_predictions):
        """Test that evaluation function returns a dictionary"""
        y_true, y_pred, y_pred_proba = sample_predictions
        result = simple_evaluate(y_true, y_pred, y_pred_proba)
        assert isinstance(result, dict)
    
    def test_returns_expected_metrics(self, sample_predictions):
        """Test that evaluation returns expected metric keys"""
        y_true, y_pred, y_pred_proba = sample_predictions
        result = simple_evaluate(y_true, y_pred, y_pred_proba)
        
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for key in expected_keys:
            assert key in result, f"Key '{key}' missing from evaluation results"
    
    def test_metric_values_in_range(self, sample_predictions):
        """Test that all metrics are between 0 and 1"""
        y_true, y_pred, y_pred_proba = sample_predictions
        result = simple_evaluate(y_true, y_pred, y_pred_proba)
        
        for key, value in result.items():
            assert 0 <= value <= 1, f"Metric '{key}' has value {value} outside [0,1] range"
    
    def test_perfect_predictions(self):
        """Test that perfect predictions yield score of 1.0"""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_pred_proba = np.array([1, 0, 1, 0, 1])
        
        result = simple_evaluate(y_true, y_pred, y_pred_proba)
        
        assert result['accuracy'] == 1.0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0
        assert result['roc_auc'] == 1.0


# ============================================
# TEST 2: CONFIDENCE SCORES
# ============================================

class TestConfidenceScores:
    """Tests for model confidence/prediction scores"""
    
    def test_probabilities_between_zero_and_one(self):
        """Test that prediction probabilities are between 0 and 1"""
        model_path = 'models/best_model_production.pkl'
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            dummy_input = pd.DataFrame({
                'TotalTransactionAmount': [1000],
                'AverageTransactionAmount': [500],
                'TransactionCount': [10],
                'StdDevTransactionAmount': [100],
                'Recency': [30]
            })
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(dummy_input)
                assert 0 <= proba[0][0] <= 1
                assert 0 <= proba[0][1] <= 1
        else:
            pytest.skip("Model file not found - skipping test")


# ============================================
# TEST 3: PREDICTION SHAPE
# ============================================

class TestPredictionShape:
    """Tests that predictions have correct shape"""
    
    def test_prediction_matches_input_length(self):
        """Test that number of predictions matches number of input samples"""
        model_path = 'models/best_model_production.pkl'
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            n_samples = 5
            dummy_input = pd.DataFrame({
                'TotalTransactionAmount': [1000] * n_samples,
                'AverageTransactionAmount': [500] * n_samples,
                'TransactionCount': [10] * n_samples,
                'StdDevTransactionAmount': [100] * n_samples,
                'Recency': [30] * n_samples
            })
            
            predictions = model.predict(dummy_input)
            assert len(predictions) == n_samples
        else:
            pytest.skip("Model file not found - skipping test")


# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])