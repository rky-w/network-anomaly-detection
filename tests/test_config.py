"""Tests for config module"""
import pytest
from network_anomaly_detection.config import config


class TestConfig:
    """Test configuration settings"""

    def test_config_exists(self):
        """Test that config dict exists and is not empty"""
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_config_has_required_keys(self):
        """Test that all required configuration keys are present"""
        required_keys = [
            'persist_window',
            'persist_threshold',
            'seasonal_freq',
            'seasonal_threshold',
            'level_shift_window',
            'level_shift_threshold',
            'pelt_penalty',
            'pelt_seasonal_decomposition',
            'isolation_forest_contamination'
        ]
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"

    def test_config_values_are_valid_types(self):
        """Test that config values have expected types"""
        assert isinstance(config['persist_window'], int)
        assert isinstance(config['persist_threshold'], int)
        assert isinstance(config['seasonal_freq'], int)
        assert isinstance(config['seasonal_threshold'], (int, float))
        assert isinstance(config['level_shift_window'], int)
        assert isinstance(config['level_shift_threshold'], (int, float))
        assert isinstance(config['pelt_penalty'], int)
        assert isinstance(config['pelt_seasonal_decomposition'], bool)
        assert isinstance(config['isolation_forest_contamination'], float)

    def test_config_values_are_positive(self):
        """Test that numeric config values are positive"""
        positive_keys = [
            'persist_window',
            'persist_threshold',
            'seasonal_freq',
            'seasonal_threshold',
            'level_shift_window',
            'level_shift_threshold',
            'pelt_penalty',
            'isolation_forest_contamination'
        ]
        for key in positive_keys:
            assert config[key] > 0, f"Config value for {key} should be positive"

    def test_contamination_is_valid_probability(self):
        """Test that isolation_forest_contamination is between 0 and 1"""
        assert 0 < config['isolation_forest_contamination'] < 1
