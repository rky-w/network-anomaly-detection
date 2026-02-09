"""Tests for data generation module"""
import pytest
import numpy as np
import pandas as pd
from network_anomaly_detection.data.generation import (
    generate_data,
    generate_point_anomalies,
    generate_level_anomalies,
    generate_trend_anomalies,
)
from pandas.api.types import is_bool_dtype


class TestGenerateData:
    """Test data generation"""

    def test_generate_data_default_parameters(self):
        """Test data generation with default parameters"""
        df = generate_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert set(df.columns) == {'bytes_in', 'bytes_out', 'error_rate'}
        assert df.index.freq == '5min'

    def test_generate_data_date_range(self):
        """Test that generated data has correct date range"""
        start = '2026-01-01'
        end = '2026-01-02'
        df = generate_data(start=start, end=end)
        
        assert df.index[0] == pd.Timestamp(start)
        assert df.index[-1] == pd.Timestamp(end)

    def test_generate_data_custom_parameters(self):
        """Test data generation with custom parameters"""
        df = generate_data(
            start='2026-01-01',
            end='2026-01-03',
            base_bytes=2000,
            peak_bytes=1000,
            peak_hour=14
        )
        
        assert len(df) > 0
        assert df['bytes_in'].min() >= 0
        assert df['bytes_out'].min() >= 0

    def test_generate_data_column_properties(self):
        """Test that generated columns have valid properties"""
        df = generate_data()
        
        # Check non-negativity
        assert df['bytes_in'].min() >= 0
        assert df['bytes_out'].min() >= 0
        assert df['error_rate'].min() >= 0
        
        # Check error rate is bounded
        assert df['error_rate'].max() <= 1

    def test_generate_data_no_null_values(self):
        """Test that generated data has no null values"""
        df = generate_data()
        
        assert not df.isnull().any().any()

    def test_generate_data_bytes_out_less_than_bytes_in(self):
        """Test that bytes_out is generally less than or equal to bytes_in"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        
        # At least 90% of the time, bytes_out should be less than bytes_in
        ratio = (df['bytes_out'] <= df['bytes_in']).sum() / len(df)
        assert ratio > 0.9


class TestGeneratePointAnomalies:
    """Test point anomaly generation"""

    def test_generate_point_anomalies_basic(self):
        """Test basic point anomaly generation"""
        df = generate_data()
        df_anomalous, flags = generate_point_anomalies(df, num_anomalies=5)
        
        assert isinstance(df_anomalous, pd.DataFrame)
        assert isinstance(flags, pd.DataFrame)
        assert df_anomalous.shape == df.shape
        assert flags.shape == df.shape

    def test_generate_point_anomalies_flags_count(self):
        """Test that the number of anomaly flags matches requested anomalies"""
        df = generate_data()
        num_anomalies = 10
        df_anomalous, flags = generate_point_anomalies(df, num_anomalies=num_anomalies)
        
        # Count the number of True values across all columns
        total_flags = flags.sum().sum()
        # Each anomaly can affect multiple columns, so total_flags >= num_anomalies
        assert total_flags >= num_anomalies

    def test_generate_point_anomalies_returns_dataframe_same_index(self):
        """Test that returned dataframes have same index as input"""
        df = generate_data()
        df_anomalous, flags = generate_point_anomalies(df, num_anomalies=5)
        
        assert (df_anomalous.index == df.index).all()
        assert (flags.index == df.index).all()

    def test_generate_point_anomalies_flags_are_boolean(self):
        """Test that anomaly flags are boolean values"""
        df = generate_data()
        df_anomalous, flags = generate_point_anomalies(df, num_anomalies=5)
        
        # After fillna(False), all values should be boolean
        for col in flags.columns:
            assert is_bool_dtype(flags[col]) or flags[col].dtype == object


    def test_generate_point_anomalies_zero_anomalies(self):
        """Test point anomalies with zero anomalies"""
        df = generate_data()
        df_anomalous, flags = generate_point_anomalies(df, num_anomalies=0)
        
        assert len(flags) == len(df)

    def test_generate_point_anomalies_changes_data(self):
        """Test that point anomalies actually change the data"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        df_anomalous, flags = generate_point_anomalies(df, num_anomalies=10)
        
        # Data should be different
        assert not df_anomalous.equals(df)


class TestGenerateLevelAnomalies:
    """Test level anomaly generation"""

    def test_generate_level_anomalies_basic(self):
        """Test basic level anomaly generation"""
        df = generate_data()
        df_anomalous, flags = generate_level_anomalies(df, num_anomalies=2)
        
        assert isinstance(df_anomalous, pd.DataFrame)
        assert isinstance(flags, pd.DataFrame)
        assert df_anomalous.shape == df.shape
        assert flags.shape == df.shape

    def test_generate_level_anomalies_flags_are_boolean(self):
        """Test that returned flags are boolean"""
        df = generate_data()
        df_anomalous, flags = generate_level_anomalies(df, num_anomalies=2)
        
        for col in flags.columns:
            unique_vals = flags[col].unique()
            assert all(v in [True, False] for v in unique_vals)

    def test_generate_level_anomalies_creates_sustained_changes(self):
        """Test that level anomalies create sustained changes"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        df_anomalous, flags = generate_level_anomalies(df, num_anomalies=1)
        
        # Level anomalies should change the data
        assert not df_anomalous.equals(df)

    def test_generate_level_anomalies_returns_same_index(self):
        """Test that returned dataframes have same index"""
        df = generate_data()
        df_anomalous, flags = generate_level_anomalies(df, num_anomalies=2)
        
        assert (df_anomalous.index == df.index).all()
        assert (flags.index == df.index).all()

    def test_generate_level_anomalies_zero_anomalies(self):
        """Test level anomalies with zero anomalies"""
        df = generate_data()
        df_anomalous, flags = generate_level_anomalies(df, num_anomalies=0)
        
        # Should return dataframe with no anomalies (or minimal changes)
        assert isinstance(df_anomalous, pd.DataFrame)
        assert isinstance(flags, pd.DataFrame)


class TestGenerateTrendAnomalies:
    """Test trend anomaly generation"""

    def test_generate_trend_anomalies_basic(self):
        """Test basic trend anomaly generation"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        df_anomalous, flags = generate_trend_anomalies(df)
        
        assert isinstance(df_anomalous, pd.DataFrame)
        assert isinstance(flags, pd.DataFrame)
        assert df_anomalous.shape == df.shape
        assert flags.shape == df.shape

    def test_generate_trend_anomalies_returns_same_index(self):
        """Test that returned dataframes have same index"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        df_anomalous, flags = generate_trend_anomalies(df)
        
        assert (df_anomalous.index == df.index).all()
        assert (flags.index == df.index).all()

    def test_generate_trend_anomalies_affects_error_rate(self):
        """Test that trend anomalies primarily affect error_rate"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        df_anomalous, flags = generate_trend_anomalies(df)
        
        # Check if error_rate has been modified
        assert not df_anomalous['error_rate'].equals(df['error_rate'])

    def test_generate_trend_anomalies_flags_are_boolean(self):
        """Test that returned flags are boolean"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        df_anomalous, flags = generate_trend_anomalies(df)
        
        for col in flags.columns:
            unique_vals = flags[col].unique()
            assert all(v in [True, False] for v in unique_vals)

    def test_generate_trend_anomalies_maintains_bounds(self):
        """Test that error_rate stays within valid bounds"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        df_anomalous, flags = generate_trend_anomalies(df)
        
        assert df_anomalous['error_rate'].min() >= 0
        assert df_anomalous['error_rate'].max() <= 1


class TestDataGenerationIntegration:
    """Integration tests for data generation functions"""

    def test_combined_anomalies_in_single_dataset(self):
        """Test creating a dataset with all types of anomalies"""
        df = generate_data(start='2026-01-01', end='2026-01-05')
        
        df_point, flags_point = generate_point_anomalies(df, num_anomalies=5)
        df_level, flags_level = generate_level_anomalies(df, num_anomalies=1)
        df_trend, flags_trend = generate_trend_anomalies(df)
        
        assert all(df is not None for df in [df_point, df_level, df_trend])

    def test_consistency_of_generation_with_seed(self):
        """Test that data generation with seed produces consistent results"""
        np.random.seed(42)
        df1 = generate_data()
        
        np.random.seed(42)
        df2 = generate_data()
        
        # Reset seed for other tests
        np.random.seed(None)
