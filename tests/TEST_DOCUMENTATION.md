# Pytest Tests for Network Anomaly Detection Project

This document provides a comprehensive overview of all pytest tests created for the network-anomaly-detection project.

## Test Files Overview

### 1. `test_config.py`
Tests for the configuration module that validates all configuration settings.

**Tests included:**
- `test_config_exists()` - Verifies config dict exists and is not empty
- `test_config_has_required_keys()` - Ensures all required keys are present
- `test_config_values_are_valid_types()` - Validates data types of configuration values
- `test_config_values_are_positive()` - Checks that numeric values are positive
- `test_contamination_is_valid_probability()` - Validates probability bounds for contamination parameter

**Key Coverage:**
- All 9 configuration parameters validated
- Type checking for int, float, and bool values
- Range validation for probability values

---

### 2. `test_data_generation.py`
Comprehensive tests for data generation functions used to create synthetic network traffic data with various types of anomalies.

#### TestGenerateData (5 tests)
- `test_generate_data_default_parameters()` - Tests basic functionality with defaults
- `test_generate_data_date_range()` - Validates date range correctness
- `test_generate_data_custom_parameters()` - Tests custom parameter values
- `test_generate_data_column_properties()` - Validates column value ranges
- `test_generate_data_no_null_values()` - Ensures no missing data
- `test_generate_data_bytes_out_less_than_bytes_in()` - Validates data relationships

#### TestGeneratePointAnomalies (6 tests)
- `test_generate_point_anomalies_basic()` - Basic functionality test
- `test_generate_point_anomalies_flags_count()` - Validates anomaly flag counts
- `test_generate_point_anomalies_returns_dataframe_same_index()` - Index preservation
- `test_generate_point_anomalies_flags_are_boolean()` - Boolean type validation
- `test_generate_point_anomalies_zero_anomalies()` - Edge case with no anomalies
- `test_generate_point_anomalies_changes_data()` - Validates data modification

#### TestGenerateLevelAnomalies (5 tests)
- `test_generate_level_anomalies_basic()` - Basic functionality
- `test_generate_level_anomalies_flags_are_boolean()` - Type validation
- `test_generate_level_anomalies_creates_sustained_changes()` - Verifies level shifts
- `test_generate_level_anomalies_returns_same_index()` - Index preservation
- `test_generate_level_anomalies_zero_anomalies()` - Zero anomalies edge case

#### TestGenerateTrendAnomalies (5 tests)
- `test_generate_trend_anomalies_basic()` - Basic functionality
- `test_generate_trend_anomalies_returns_same_index()` - Index preservation
- `test_generate_trend_anomalies_affects_error_rate()` - Validates trend changes
- `test_generate_trend_anomalies_flags_are_boolean()` - Type validation
- `test_generate_trend_anomalies_maintains_bounds()` - Bounds checking

#### TestDataGenerationIntegration (2 tests)
- `test_combined_anomalies_in_single_dataset()` - Tests combining multiple anomaly types
- `test_consistency_of_generation_with_seed()` - Tests reproducibility with seeds

**Total: 23 tests for data generation**

---

### 3. `test_utils.py`
Tests for utility functions used in anomaly analysis.

#### TestCombineAnomalies (10 tests)
- `test_combine_anomalies_with_two_dataframes()` - Basic OR logic for two dataframes
- `test_combine_anomalies_with_multiple_dataframes()` - OR logic with 3+ dataframes
- `test_combine_anomalies_with_single_dataframe()` - Single dataframe handling
- `test_combine_anomalies_all_false()` - All False values case
- `test_combine_anomalies_all_true()` - All True values case
- `test_combine_anomalies_multiple_columns()` - Multi-column OR logic
- `test_combine_anomalies_preserves_index()` - Index preservation
- `test_combine_anomalies_result_is_dataframe()` - Return type validation
- `test_combine_anomalies_with_mixed_boolean_and_numeric()` - Type flexibility
- `test_combine_anomalies_with_large_dataset()` - Scalability test with 1000 rows

**Total: 10 tests for utilities**

---

### 4. `test_detectors.py`
Comprehensive tests for anomaly detection algorithms (CUSUM and PELT).

#### TestCUSUMAD (11 tests)
- `test_cusumad_initialization_defaults()` - Default parameter test
- `test_cusumad_initialization_custom_parameters()` - Custom parameter test
- `test_cusumad_fit_returns_self()` - Method chaining support
- `test_cusumad_fit_stores_standardization_params()` - Parameter storage validation
- `test_cusumad_predict_before_fit_raises_error()` - Error handling
- `test_cusumad_predict_returns_dataframe()` - Return type validation
- `test_cusumad_predict_returns_boolean_values()` - Boolean output validation
- `test_cusumad_predict_has_correct_shape()` - Shape preservation
- `test_cusumad_predict_has_correct_columns()` - Column preservation
- `test_cusumad_predict_has_correct_index()` - Index preservation
- `test_cusumad_fit_predict()` - Combined fit/predict method
- `test_cusumad_different_thresholds_produce_different_results()` - Parameter sensitivity
- `test_cusumad_with_seasonal_decomposition()` - Optional feature support
- `test_cusumad_refit_updates_parameters()` - Refit capability

#### TestPELTAD (14 tests)
- Similar comprehensive tests as CUSUMAD, including:
  - Initialization (default and custom)
  - Fit and predict functionality
  - Parameter storage
  - Error handling
  - Output validation (shape, columns, index, type)
  - Method chaining
  - fit_predict() combined method
  - Parameter sensitivity (penalty levels)
  - Seasonal decomposition support
  - Refit capability

#### TestDetectorComparison (2 tests)
- `test_both_detectors_work_on_same_data()` - Cross-compatibility test
- `test_detectors_on_clean_data_produce_few_anomalies()` - Expected behavior

#### TestDetectorEdgeCases (3 tests)
- `test_detector_with_constant_data()` - Constant data handling
- `test_detector_with_very_small_dataset()` - Minimal data support
- `test_detector_with_nans_handled()` - NaN handling validation

**Total: 34 tests for detectors**

---

## Running the Tests

### Quick Start
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test file
pytest tests/test_config.py

# Run specific test class
pytest tests/test_detectors.py::TestCUSUMAD

# Run specific test
pytest tests/test_config.py::TestConfig::test_config_exists

# Run with coverage report
pytest --cov=src/network_anomaly_detection tests/

# Run with detailed output and print statements
pytest -vv -s tests/
```

### Installation Requirements
Ensure pytest is installed:
```bash
pip install pytest pytest-cov
```

Or install the development dependencies:
```bash
pip install -e ".[dev]"  # if dev extras are configured
```

---

## Test Statistics

| Module | Test File | Test Count |
|--------|-----------|-----------|
| config | test_config.py | 5 |
| data generation | test_data_generation.py | 23 |
| utilities | test_utils.py | 10 |
| detectors | test_detectors.py | 34 |
| **Total** | | **72** |

---

## Test Coverage Areas

### Functionality Coverage
- ✅ Configuration validation
- ✅ Data generation (point, level, and trend anomalies)
- ✅ Anomaly combination logic
- ✅ CUSUM anomaly detection
- ✅ PELT changepoint detection
- ✅ Seasonal decomposition
- ✅ Standardization and preprocessing

### Edge Cases Covered
- ✅ Empty/zero parameters
- ✅ Single element inputs
- ✅ Large datasets (1000+ rows)
- ✅ Constant data
- ✅ Minimum viable data
- ✅ Parameter combinations
- ✅ Refit scenarios

### Quality Checks
- ✅ Return type validation
- ✅ Shape and index preservation
- ✅ Value range validation
- ✅ Boundary conditions
- ✅ Error handling
- ✅ Boolean type correctness
- ✅ Null value handling

---

## Fixtures

The `conftest.py` file provides pytest fixtures:

### `sample_dataframe`
Provides a clean sample network traffic dataframe with 24 hours of 5-minute intervals.

### `anomaly_dataframe`
Provides a dataframe with intentional point anomalies and level shifts.

---

## Notes

1. **Dependencies**: Tests require pandas, numpy, scikit-learn, ruptures, and adtk
2. **Random Seed**: Some tests use `np.random.seed(42)` for reproducibility
3. **Performance**: All tests should complete in seconds
4. **Isolation**: Tests are independent and can run in any order
5. **Fixtures**: Reusable test data fixtures in conftest.py

---

## Future Test Enhancements

Potential areas for additional testing:
- Integration tests combining multiple modules
- Performance benchmarks for large datasets
- Visualization output validation
- Concurrent detector execution
- GPU acceleration support (if implemented)
- Cross-validation scoring
- Hyperparameter optimization validation
