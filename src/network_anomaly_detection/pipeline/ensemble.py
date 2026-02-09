
from adtk.data import validate_series
from adtk.detector import LevelShiftAD, PersistAD, SeasonalAD
from network_anomaly_detection.utils import combine_anomalies
from network_anomaly_detection.detectors import PELTAD
from network_anomaly_detection.config import config

from adtk.data import validate_series


def run_ensemble_pipeline(df):
    """Run the full ensemble anomaly detection pipeline"""

    # Check for consistency and absence on duplicates in the timeseries
    df = validate_series(df)

    # Run detectors
    persist_ad = PersistAD(window=config['persist_window'], c=config['persist_threshold'])
    persist_anomalies = persist_ad.fit_predict(df)

    seasonal_ad = SeasonalAD(freq=config['seasonal_freq'], c=config['seasonal_threshold'])
    seasonal_anomalies = seasonal_ad.fit_predict(df)

    level_shift_ad = LevelShiftAD(window=config['level_shift_window'], c=config['level_shift_threshold'])
    level_anomalies = level_shift_ad.fit_predict(df)

    pelt_ad = PELTAD(penalty=config['pelt_penalty'], seasonal_decomposition=config['pelt_seasonal_decomposition'])
    pelt_anomalies = pelt_ad.fit_predict(df)                                                                            


    # Combine anomalies using OR logic
    combined_anomalies = combine_anomalies([
        persist_anomalies,
        seasonal_anomalies,
        level_anomalies,
        pelt_anomalies
        ])

    return combined_anomalies

