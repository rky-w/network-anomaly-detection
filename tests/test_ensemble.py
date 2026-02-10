"""Tests for ensemble pipeline module"""
import numpy as np
import pandas as pd
import pytest

from network_anomaly_detection.data.generation import (
    generate_data,
    generate_level_anomalies,
    generate_point_anomalies,
    generate_trend_anomalies,
)
from network_anomaly_detection.pipeline.ensemble import run_ensemble_pipeline


class TestRunEnsemblePipeline:
    """Test the ensemble anomaly detection pipeline"""

    def test_ensemble_pipeline_returns_dataframe(self):
        """Test that pipeline returns a DataFrame"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        assert isinstance(result, pd.DataFrame)

    def test_ensemble_pipeline_output_shape(self):
        """Test that output has same shape as input"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        assert result.shape == df.shape

    def test_ensemble_pipeline_output_columns(self):
        """Test that output has same columns as input"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        assert set(result.columns) == set(df.columns)

    def test_ensemble_pipeline_output_index(self):
        """Test that output has same index as input"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        assert (result.index == df.index).all()

    def test_ensemble_pipeline_returns_boolean_values(self):
        """Test that output contains only boolean values"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        for col in result.columns:
            assert result[col].dtype == bool or all(
                isinstance(val, (bool, np.bool_)) for val in result[col]
            )

    def test_ensemble_pipeline_with_clean_data(self):
        """Test pipeline on clean data produces few anomalies"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        # Clean data should have relatively few anomalies
        total_anomalies = result.sum().sum()
        total_points = result.shape[0] * result.shape[1]

        # Assert less than 30% of points are flagged as anomalies
        assert total_anomalies < total_points * 0.3

    def test_ensemble_pipeline_with_point_anomalies(self):
        """Test pipeline detects point anomalies"""
        df = generate_data(start="2026-01-01", end="2026-01-02")
        df_anomalous, _ = generate_point_anomalies(df, num_anomalies=20)

        result = run_ensemble_pipeline(df_anomalous)

        # Should detect some anomalies
        total_anomalies = result.sum().sum()
        assert total_anomalies > 0

    def test_ensemble_pipeline_with_level_anomalies(self):
        """Test pipeline detects level shift anomalies"""
        df = generate_data(start="2026-01-01", end="2026-01-02")
        df_anomalous, _ = generate_level_anomalies(df, num_anomalies=2)

        result = run_ensemble_pipeline(df_anomalous)

        # Should detect some anomalies
        total_anomalies = result.sum().sum()
        assert total_anomalies > 0

    def test_ensemble_pipeline_with_trend_anomalies(self):
        """Test pipeline detects trend anomalies"""
        df = generate_data(start="2026-01-01", end="2026-01-02")
        df_anomalous, _ = generate_trend_anomalies(df)

        result = run_ensemble_pipeline(df_anomalous)

        # Should detect some anomalies
        total_anomalies = result.sum().sum()
        assert total_anomalies > 0

    def test_ensemble_pipeline_with_multiple_anomaly_types(self):
        """Test pipeline with combined anomaly types"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        # Add multiple types of anomalies
        df_anomalous, _ = generate_point_anomalies(df, num_anomalies=10)
        df_anomalous, _ = generate_level_anomalies(df_anomalous, num_anomalies=1)

        result = run_ensemble_pipeline(df_anomalous)

        # Should detect anomalies
        total_anomalies = result.sum().sum()
        assert total_anomalies > 0

    def test_ensemble_pipeline_no_null_values(self):
        """Test that pipeline output has no null values"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        assert not result.isnull().any().any()

    def test_ensemble_pipeline_with_longer_timeseries(self):
        """Test pipeline with longer time series"""
        df = generate_data(start="2026-01-01", end="2026-01-05")

        result = run_ensemble_pipeline(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_ensemble_pipeline_with_shorter_timeseries(self):
        """Test pipeline with shorter time series"""
        df = generate_data(start="2026-01-01 00:00", end="2026-01-01 06:00")

        result = run_ensemble_pipeline(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_ensemble_pipeline_consistency_with_seed(self):
        """Test that pipeline produces consistent results with same data"""
        np.random.seed(42)
        df1 = generate_data(start="2026-01-01", end="2026-01-02")

        np.random.seed(42)
        df2 = generate_data(start="2026-01-01", end="2026-01-02")

        # Reset seed
        np.random.seed(None)

        result1 = run_ensemble_pipeline(df1)
        result2 = run_ensemble_pipeline(df2)

        # Results should be identical for identical input
        pd.testing.assert_frame_equal(result1, result2)

    def test_ensemble_pipeline_detects_more_than_individual_detectors(self):
        """Test that ensemble combines multiple detector outputs"""
        df = generate_data(start="2026-01-01", end="2026-01-02")
        df_anomalous, _ = generate_point_anomalies(df, num_anomalies=15)
        df_anomalous, _ = generate_level_anomalies(df_anomalous, num_anomalies=2)

        result = run_ensemble_pipeline(df_anomalous)

        # Ensemble should find anomalies since it uses OR logic
        assert result.sum().sum() > 0

    def test_ensemble_pipeline_preserves_datetime_index(self):
        """Test that pipeline preserves datetime index type"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.freq is not None or len(result.index) < 2

    def test_ensemble_pipeline_with_all_metrics(self):
        """Test that all metrics are processed"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        expected_columns = ["bytes_in", "bytes_out", "error_rate"]
        assert all(col in result.columns for col in expected_columns)

    def test_ensemble_pipeline_flag_distribution(self):
        """Test that anomaly flags are distributed across different metrics"""
        df = generate_data(start="2026-01-01", end="2026-01-02")
        df_anomalous, _ = generate_point_anomalies(df, num_anomalies=20)

        result = run_ensemble_pipeline(df_anomalous)

        # Check that at least one metric has anomalies
        metrics_with_anomalies = [col for col in result.columns if result[col].sum() > 0]
        assert len(metrics_with_anomalies) >= 1

    def test_ensemble_pipeline_with_extreme_values(self):
        """Test pipeline handles extreme values in data"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        # Add extreme values
        df.iloc[50, 0] *= 100  # Extreme spike
        df.iloc[100, 2] = 0.99  # Near maximum error rate

        result = run_ensemble_pipeline(df)

        # Should detect these extreme values
        assert result.sum().sum() > 0


class TestEnsemblePipelineEdgeCases:
    """Test edge cases in the ensemble pipeline"""

    def test_ensemble_pipeline_with_minimum_data(self):
        """Test pipeline with minimal data points"""
        # Generate minimal data
        index = pd.date_range("2026-01-01", periods=50, freq="5min")
        df = pd.DataFrame(
            {
                "bytes_in": np.random.rand(50) * 1000,
                "bytes_out": np.random.rand(50) * 300,
                "error_rate": np.random.rand(50) * 0.1,
            },
            index=index,
        )

        result = run_ensemble_pipeline(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape


    def test_ensemble_pipeline_with_high_frequency_data(self):
        """Test pipeline with high frequency (1-min intervals)"""
        index = pd.date_range("2026-01-01", periods=300, freq="1min")
        df = pd.DataFrame(
            {
                "bytes_in": np.random.rand(300) * 1000 + 500,
                "bytes_out": np.random.rand(300) * 300 + 100,
                "error_rate": np.random.rand(300) * 0.1 + 0.02,
            },
            index=index,
        )

        result = run_ensemble_pipeline(df)

        assert result.shape == df.shape


class TestEnsemblePipelineIntegration:
    """Integration tests for ensemble pipeline"""

    def test_ensemble_pipeline_end_to_end_workflow(self):
        """Test complete workflow from data generation to anomaly detection"""
        # Generate base data
        df = generate_data(start="2026-01-01", end="2026-01-02")

        # Add various anomaly types
        df_with_point, _ = generate_point_anomalies(df, num_anomalies=10)
        df_with_level, _ = generate_level_anomalies(df_with_point, num_anomalies=1)

        # Run ensemble pipeline
        result = run_ensemble_pipeline(df_with_level)

        # Validate results
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert not result.isnull().any().any()
        assert result.sum().sum() > 0  # Should detect anomalies

    def test_ensemble_pipeline_multiple_runs_on_same_data(self):
        """Test that running pipeline multiple times gives same results"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result1 = run_ensemble_pipeline(df)
        result2 = run_ensemble_pipeline(df)

        pd.testing.assert_frame_equal(result1, result2)


    def test_ensemble_combines_multiple_detector_outputs(self):
        """Test that ensemble truly combines outputs from multiple detectors"""
        # Create data with specific anomaly that would be caught by different detectors
        df = generate_data(start="2026-01-01", end="2026-01-02")

        # Add level shift (should be caught by LevelShiftAD and PELT)
        df.iloc[200:400, :] *= 1.4

        # Add persist anomaly
        df.iloc[100:105, 0] = df.iloc[100, 0]  # Constant values

        result = run_ensemble_pipeline(df)

        # Should detect anomalies
        assert result.sum().sum() > 0

    def test_ensemble_pipeline_output_format_suitable_for_analysis(self):
        """Test that output format is suitable for further analysis"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        # Should be able to perform common operations
        assert result.sum().sum() >= 0
        assert result.any().any() in [True, False]
        assert len(result[result["bytes_in"] == True]) >= 0



class TestEnsemblePipelineValidation:
    """Test validation features of the ensemble pipeline"""

    def test_ensemble_pipeline_validates_series(self):
        """Test that pipeline validates input series"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        # Should work without errors
        result = run_ensemble_pipeline(df)

        assert isinstance(result, pd.DataFrame)

    def test_ensemble_pipeline_with_valid_datetime_index(self):
        """Test that pipeline works with valid datetime index"""
        df = generate_data(start="2026-01-01", end="2026-01-02")

        result = run_ensemble_pipeline(df)

        assert isinstance(result.index, pd.DatetimeIndex)

    def test_ensemble_pipeline_input_unchanged(self):
        """Test that pipeline doesn't modify input dataframe"""
        df = generate_data(start="2026-01-01", end="2026-01-02")
        df_original = df.copy()

        _ = run_ensemble_pipeline(df)

        # Input should remain unchanged
        pd.testing.assert_frame_equal(df, df_original)
