import numpy as np
import pandas as pd
from adtk.transformer import ClassicSeasonalDecomposition


# CUSUM anomaly detection with fit/predict interface
class CUSUMAD:
    """
    CUSUM (Cumulative Sum) anomaly detector with fit/predict interface.
    
    CUSUM detects trend changes by accumulating deviations from the mean.
    For each sequential group of anomaly flags, only the first index is marked as True.
    
    Note: CUSUM is an unsupervised algorithm that analyzes data independently to find
    trend changes. What we "fit" is the preprocessing (standardization and 
    deseasonalization) so it's consistent between train and test data. The CUSUM algorithm
    itself runs fresh on each predict() call, which is the intended behavior for 
    unsupervised anomaly detection.
    """
    
    def __init__(self, threshold=5, seasonal_decomposition=False, freq=288):
        """
        Parameters
        ----------
        threshold : float, optional
            Threshold for cumulative sum. Higher values result in fewer detected anomalies.
        seasonal_decomposition : bool, optional
            Whether to deseasonalize the data before detection.
        freq : int, optional
            Frequency for seasonal decomposition (e.g., 288 for daily seasonality with 5-min intervals).
        """
        self.threshold = threshold
        self.seasonal_decomposition = seasonal_decomposition
        self.freq = freq
        self.means = {}
        self.stds = {}
        self.deseasonalisers = {}
        self.is_fitted = False
    
    def fit(self, df):
        """
        Fit the detector to training data.
        Stores standardization parameters and fits deseasonalizers.
        """
        for col in df.columns:
            series = df[col]
            
            # Store standardization parameters
            self.means[col] = np.mean(series)
            self.stds[col] = np.std(series)
            
            # Standardize the series
            series_standardized = (series - self.means[col]) / self.stds[col]
            
            # Fit and store deseasonalizer if needed
            if self.seasonal_decomposition:
                deseasonaliser = ClassicSeasonalDecomposition(freq=self.freq)
                deseasonaliser.fit(series_standardized)
                self.deseasonalisers[col] = deseasonaliser
        
        self.is_fitted = True
        return self
    
    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before calling predict(). Call fit() first.")
        
        anomalies = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            series = df[col].copy()
            
            # Apply standardization using stored parameters from fit()
            series = (series - self.means[col]) / self.stds[col]
            
            # Apply deseasonalization if fitted
            if self.seasonal_decomposition:
                series = self.deseasonalisers[col].transform(series)
            
            # CUSUM analyzes this preprocessed data to find trend changes
            cumsum = np.cumsum(series - series.mean())
            anomaly_flags = np.abs(cumsum) > self.threshold
            
            # Keep only the first index of each sequential group of anomalies
            anomalies_series = pd.Series(anomaly_flags, index=series.index)
            changes = anomalies_series.astype(int).diff().fillna(0)
            # Transitions from False to True (value of 1) indicate the start of anomaly groups
            pred_anomalies = (changes == 1)
            
            anomalies[col] = pred_anomalies
        
        return anomalies
    
    def fit_predict(self, df):
        return self.fit(df).predict(df)

