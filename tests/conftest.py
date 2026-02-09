"""Pytest configuration file"""
import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample network traffic dataframe"""
    import pandas as pd
    import numpy as np
    
    index = pd.date_range('2026-01-01', periods=288, freq='5min')
    df = pd.DataFrame({
        'bytes_in': np.random.rand(288) * 1000 + 500,
        'bytes_out': np.random.rand(288) * 300 + 100,
        'error_rate': np.random.rand(288) * 0.1 + 0.02,
    }, index=index)
    return df


@pytest.fixture
def anomaly_dataframe(sample_dataframe):
    """Fixture providing a dataframe with anomalies"""
    df = sample_dataframe.copy()
    
    # Add some point anomalies
    df.iloc[50, 0] *= 5  # Spike in bytes_in
    df.iloc[100, 2] = 0.95  # Spike in error_rate
    
    # Add a level change
    df.iloc[150:200, :] *= 1.3
    
    return df
