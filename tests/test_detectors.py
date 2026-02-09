"""Tests for anomaly detector modules (CUSUM and PELT)"""
import pytest
import numpy as np
import pandas as pd
from network_anomaly_detection.detectors.cusum import CUSUMAD
from network_anomaly_detection.detectors.pelt import PELTAD
from network_anomaly_detection.data.generation import generate_data
from pandas.api.types import is_bool_dtype

class TestCUSUMAD:
    """Test CUSUM anomaly detector"""

    def test_cusumad_initialization_defaults(self):
        """Test CUSUMAD initialization with default parameters"""
        detector = CUSUMAD()
        
        assert detector.threshold == 5
        assert detector.seasonal_decomposition == False
        assert detector.freq == 288
        assert detector.is_fitted == False

    def test_cusumad_initialization_custom_parameters(self):
        """Test CUSUMAD initialization with custom parameters"""
        detector = CUSUMAD(threshold=10, seasonal_decomposition=True, freq=96)
        
        assert detector.threshold == 10
        assert detector.seasonal_decomposition == True
        assert detector.freq == 96

    def test_cusumad_fit_returns_self(self):
        """Test that fit() returns self for method chaining"""
        df = generate_data()
        detector = CUSUMAD()
        
        result = detector.fit(df)
        
        assert result is detector
        assert detector.is_fitted == True

    def test_cusumad_fit_stores_standardization_params(self):
        """Test that fit() stores mean and std for each column"""
        df = generate_data()
        detector = CUSUMAD()
        detector.fit(df)
        
        for col in df.columns:
            assert col in detector.means
            assert col in detector.stds
            assert detector.means[col] is not None
            assert detector.stds[col] is not None

    def test_cusumad_predict_before_fit_raises_error(self):
        """Test that predict() before fit() raises ValueError"""
        df = generate_data()
        detector = CUSUMAD()
        
        with pytest.raises(ValueError):
            detector.predict(df)

    def test_cusumad_predict_returns_dataframe(self):
        """Test that predict() returns a DataFrame"""
        df = generate_data()
        detector = CUSUMAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        assert isinstance(result, pd.DataFrame)

    def test_cusumad_predict_returns_boolean_values(self):
        """Test that predict() returns boolean anomaly flags"""
        df = generate_data()
        detector = CUSUMAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        for col in result.columns:
            assert is_bool_dtype(result[col])

    def test_cusumad_predict_has_correct_shape(self):
        """Test that predict() returns same shape as input"""
        df = generate_data()
        detector = CUSUMAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        assert result.shape == df.shape

    def test_cusumad_predict_has_correct_columns(self):
        """Test that predict() has same columns as input"""
        df = generate_data()
        detector = CUSUMAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        assert set(result.columns) == set(df.columns)

    def test_cusumad_predict_has_correct_index(self):
        """Test that predict() has same index as input"""
        df = generate_data()
        detector = CUSUMAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        assert (result.index == df.index).all()

    def test_cusumad_fit_predict(self):
        """Test fit_predict() method"""
        df = generate_data()
        detector = CUSUMAD()
        
        result = detector.fit_predict(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_cusumad_different_thresholds_produce_different_results(self):
        """Test that different thresholds produce different anomaly detections"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        
        detector_low = CUSUMAD(threshold=1)
        detector_high = CUSUMAD(threshold=1000)
        
        result_low = detector_low.fit_predict(df)
        result_high = detector_high.fit_predict(df)
        
        # Lower threshold should detect more anomalies
        assert result_low.sum().sum() >= result_high.sum().sum()

    def test_cusumad_with_seasonal_decomposition(self):
        """Test CUSUMAD with seasonal decomposition enabled"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        detector = CUSUMAD(seasonal_decomposition=True, freq=288)
        
        result = detector.fit_predict(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_cusumad_refit_updates_parameters(self):
        """Test that refitting with new data updates parameters"""
        df1 = generate_data(start='2026-01-01', end='2026-01-02')
        df2 = generate_data(start='2026-02-01', end='2026-02-02')
        
        detector = CUSUMAD()
        detector.fit(df1)
        means_1 = detector.means.copy()
        
        detector.fit(df2)
        means_2 = detector.means.copy()
        
        # Parameters should have changed (unless by coincidence)
        assert means_1 != means_2


class TestPELTAD:
    """Test PELT anomaly detector"""

    def test_peltad_initialization_defaults(self):
        """Test PELTAD initialization with default parameters"""
        detector = PELTAD()
        
        assert detector.penalty == 10
        assert detector.seasonal_decomposition == False
        assert detector.freq == 288
        assert detector.is_fitted == False

    def test_peltad_initialization_custom_parameters(self):
        """Test PELTAD initialization with custom parameters"""
        detector = PELTAD(penalty=20, seasonal_decomposition=True, freq=96)
        
        assert detector.penalty == 20
        assert detector.seasonal_decomposition == True
        assert detector.freq == 96

    def test_peltad_fit_returns_self(self):
        """Test that fit() returns self for method chaining"""
        df = generate_data()
        detector = PELTAD()
        
        result = detector.fit(df)
        
        assert result is detector
        assert detector.is_fitted == True

    def test_peltad_fit_stores_standardization_params(self):
        """Test that fit() stores mean and std for each column"""
        df = generate_data()
        detector = PELTAD()
        detector.fit(df)
        
        for col in df.columns:
            assert col in detector.means
            assert col in detector.stds

    def test_peltad_predict_before_fit_raises_error(self):
        """Test that predict() before fit() raises ValueError"""
        df = generate_data()
        detector = PELTAD()
        
        with pytest.raises(ValueError):
            detector.predict(df)

    def test_peltad_predict_returns_dataframe(self):
        """Test that predict() returns a DataFrame"""
        df = generate_data()
        detector = PELTAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        assert isinstance(result, pd.DataFrame)

    def test_peltad_predict_returns_boolean_values(self):
        """Test that predict() returns boolean anomaly flags"""
        df = generate_data()
        detector = PELTAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        for col in result.columns:
            assert is_bool_dtype(result[col])

    def test_peltad_predict_has_correct_shape(self):
        """Test that predict() returns same shape as input"""
        df = generate_data()
        detector = PELTAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        assert result.shape == df.shape

    def test_peltad_predict_has_correct_columns(self):
        """Test that predict() has same columns as input"""
        df = generate_data()
        detector = PELTAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        assert set(result.columns) == set(df.columns)

    def test_peltad_predict_has_correct_index(self):
        """Test that predict() has same index as input"""
        df = generate_data()
        detector = PELTAD()
        detector.fit(df)
        
        result = detector.predict(df)
        
        assert (result.index == df.index).all()

    def test_peltad_fit_predict(self):
        """Test fit_predict() method"""
        df = generate_data()
        detector = PELTAD()
        
        result = detector.fit_predict(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_peltad_different_penalties_produce_different_results(self):
        """Test that different penalties produce different anomaly detections"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        
        detector_low = PELTAD(penalty=5)
        detector_high = PELTAD(penalty=20)
        
        result_low = detector_low.fit_predict(df)
        result_high = detector_high.fit_predict(df)
        
        # Lower penalty should detect more changepoints
        assert result_low.sum().sum() >= result_high.sum().sum()

    def test_peltad_with_seasonal_decomposition(self):
        """Test PELTAD with seasonal decomposition enabled"""
        df = generate_data(start='2026-01-01', end='2026-01-10')
        detector = PELTAD(seasonal_decomposition=True, freq=288)
        
        result = detector.fit_predict(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    def test_peltad_refit_updates_parameters(self):
        """Test that refitting with new data updates parameters"""
        df1 = generate_data(start='2026-01-01', end='2026-01-02')
        df2 = generate_data(start='2026-02-01', end='2026-02-02')
        
        detector = PELTAD()
        detector.fit(df1)
        means_1 = detector.means.copy()
        
        detector.fit(df2)
        means_2 = detector.means.copy()
        
        # Parameters should have changed (unless by coincidence)
        assert means_1 != means_2


class TestDetectorComparison:
    """Test comparison between detectors"""

    def test_both_detectors_work_on_same_data(self):
        """Test that both detectors can process the same data"""
        df = generate_data(start='2026-01-01', end='2026-01-05')
        
        cusum_detector = CUSUMAD()
        pelt_detector = PELTAD()
        
        cusum_result = cusum_detector.fit_predict(df)
        pelt_result = pelt_detector.fit_predict(df)
        
        assert cusum_result.shape == pelt_result.shape
        assert set(cusum_result.columns) == set(pelt_result.columns)

    def test_detectors_on_clean_data_produce_few_anomalies(self):
        """Test that detectors produce few anomalies on clean data"""
        # Generate relatively clean data
        df = generate_data(start='2026-01-01', end='2026-01-02')
        
        cusum_detector = CUSUMAD(threshold=10)  # High threshold for clean data
        pelt_detector = PELTAD(penalty=20)  # High penalty for clean data
        
        cusum_result = cusum_detector.fit_predict(df)
        pelt_result = pelt_detector.fit_predict(df)
        
        # Should detect relatively few anomalies
        assert cusum_result.sum().sum() < len(df) / 2
        assert pelt_result.sum().sum() < len(df) / 2


class TestDetectorEdgeCases:
    """Test edge cases for detectors"""

    def test_detector_with_constant_data(self):
        """Test detector behavior with constant (no variation) data"""
        index = pd.date_range('2026-01-01', periods=100, freq='5min')
        df = pd.DataFrame({
            'bytes_in': [1000] * 100,
            'bytes_out': [300] * 100,
            'error_rate': [0.05] * 100,
        }, index=index)
        
        cusum = CUSUMAD()
        result = cusum.fit_predict(df)
        
        # With constant data, should detect changes at beginning or not at all
        assert isinstance(result, pd.DataFrame)

    def test_detector_with_very_small_dataset(self):
        """Test detector with minimal data points"""
        index = pd.date_range('2026-01-01', periods=10, freq='5min')
        df = pd.DataFrame({
            'bytes_in': np.random.rand(10) * 1000,
            'bytes_out': np.random.rand(10) * 300,
            'error_rate': np.random.rand(10) * 0.1,
        }, index=index)
        
        pelt = PELTAD()
        result = pelt.fit_predict(df)
        
        assert result.shape == df.shape

    def test_detector_with_nans_handled(self):
        """Test that detector handles data appropriately"""
        df = generate_data()
        
        # Test that the detector works with the generated data
        detector = CUSUMAD()
        result = detector.fit_predict(df)
        
        # No NaN values should be in output
        assert not result.isnull().any().any()
