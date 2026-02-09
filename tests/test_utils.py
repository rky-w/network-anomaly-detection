"""Tests for utils module"""
import pytest
import numpy as np
import pandas as pd
from network_anomaly_detection.utils import combine_anomalies


class TestCombineAnomalies:
    """Test anomaly combination using OR logic"""

    def test_combine_anomalies_with_two_dataframes(self):
        """Test combining two anomaly dataframes"""
        # Create sample anomaly dataframes
        index = pd.date_range('2026-01-01', periods=5, freq='5min')
        df1 = pd.DataFrame({
            'metric_a': [True, False, True, False, False],
            'metric_b': [False, False, False, True, False],
        }, index=index)
        
        df2 = pd.DataFrame({
            'metric_a': [False, True, False, False, True],
            'metric_b': [True, False, False, False, False],
        }, index=index)
        
        result = combine_anomalies([df1, df2])
        
        # Verify OR logic: True if either df1 or df2 has True
        assert result.iloc[0]['metric_a'] == True   # True | False
        assert result.iloc[1]['metric_a'] == True   # False | True
        assert result.iloc[2]['metric_a'] == True   # True | False
        assert result.iloc[3]['metric_a'] == False  # False | False
        assert result.iloc[4]['metric_a'] == True   # False | True

    def test_combine_anomalies_with_multiple_dataframes(self):
        """Test combining more than two dataframes"""
        index = pd.date_range('2026-01-01', periods=3, freq='5min')
        
        df1 = pd.DataFrame({
            'metric': [True, False, False],
        }, index=index)
        
        df2 = pd.DataFrame({
            'metric': [False, True, False],
        }, index=index)
        
        df3 = pd.DataFrame({
            'metric': [False, False, True],
        }, index=index)
        
        result = combine_anomalies([df1, df2, df3])
        
        # All should be True (OR logic across all three)
        assert result['metric'].all()

    def test_combine_anomalies_with_single_dataframe(self):
        """Test combining a single dataframe returns a copy"""
        index = pd.date_range('2026-01-01', periods=3, freq='5min')
        df = pd.DataFrame({
            'metric': [True, False, True],
        }, index=index)
        
        result = combine_anomalies([df])
        
        assert result.equals(df)
        assert result is not df  # Should be a copy

    def test_combine_anomalies_all_false(self):
        """Test combining dataframes with all False values"""
        index = pd.date_range('2026-01-01', periods=4, freq='5min')
        df1 = pd.DataFrame({
            'metric': [False, False, False, False],
        }, index=index)
        
        df2 = pd.DataFrame({
            'metric': [False, False, False, False],
        }, index=index)
        
        result = combine_anomalies([df1, df2])
        
        assert not result['metric'].any()

    def test_combine_anomalies_all_true(self):
        """Test combining dataframes with all True values"""
        index = pd.date_range('2026-01-01', periods=3, freq='5min')
        df1 = pd.DataFrame({
            'metric': [True, True, True],
        }, index=index)
        
        df2 = pd.DataFrame({
            'metric': [True, True, True],
        }, index=index)
        
        result = combine_anomalies([df1, df2])
        
        assert result['metric'].all()

    def test_combine_anomalies_multiple_columns(self):
        """Test combining dataframes with multiple columns"""
        index = pd.date_range('2026-01-01', periods=3, freq='5min')
        df1 = pd.DataFrame({
            'metric_a': [True, False, True],
            'metric_b': [False, False, True],
            'metric_c': [True, True, False],
        }, index=index)
        
        df2 = pd.DataFrame({
            'metric_a': [False, True, False],
            'metric_b': [True, False, False],
            'metric_c': [False, True, False],
        }, index=index)
        
        result = combine_anomalies([df1, df2])
        
        # Verify each column independently
        assert result['metric_a'].tolist() == [True, True, True]
        assert result['metric_b'].tolist() == [True, False, True]
        assert result['metric_c'].tolist() == [True, True, False]

    def test_combine_anomalies_preserves_index(self):
        """Test that combining preserves the original index"""
        index = pd.date_range('2026-01-01', periods=5, freq='5min')
        df1 = pd.DataFrame({
            'metric': [True, False, True, False, True],
        }, index=index)
        
        df2 = pd.DataFrame({
            'metric': [False, True, False, True, False],
        }, index=index)
        
        result = combine_anomalies([df1, df2])
        
        assert (result.index == index).all()

    def test_combine_anomalies_result_is_dataframe(self):
        """Test that result is always a DataFrame"""
        index = pd.date_range('2026-01-01', periods=3, freq='5min')
        df = pd.DataFrame({
            'metric': [True, False, True],
        }, index=index)
        
        result = combine_anomalies([df])
        
        assert isinstance(result, pd.DataFrame)

    def test_combine_anomalies_with_mixed_boolean_and_numeric(self):
        """Test combining dataframes with different boolean representations"""
        index = pd.date_range('2026-01-01', periods=3, freq='5min')
        df1 = pd.DataFrame({
            'metric': [1, 0, 1],  # numeric representation
        }, index=index)
        
        df2 = pd.DataFrame({
            'metric': [0, 1, 0],  # numeric representation
        }, index=index)
        
        result = combine_anomalies([df1, df2])
        
        # Should use OR logic on numeric values
        assert result['metric'].tolist() == [1, 1, 1]

    def test_combine_anomalies_with_large_dataset(self):
        """Test combining with a larger dataset"""
        index = pd.date_range('2026-01-01', periods=1000, freq='5min')
        
        np.random.seed(42)
        df1 = pd.DataFrame({
            'metric': np.random.choice([True, False], size=1000),
        }, index=index)
        
        df2 = pd.DataFrame({
            'metric': np.random.choice([True, False], size=1000),
        }, index=index)
        
        result = combine_anomalies([df1, df2])
        
        # Check that result has the expected shape
        assert result.shape == df1.shape
        assert len(result) == 1000
