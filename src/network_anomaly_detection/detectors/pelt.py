import numpy as np
import pandas as pd
import ruptures as rpt
from adtk.transformer import ClassicSeasonalDecomposition


# PELT anomaly detection with fit/predict interface
class PELTAD:
    """
    PELT (Pruned Exact Linear Time) changepoint detector with fit/predict interface.    
    Note: PELT is an unsupervised algorithm that analyzes data independently to find
    optimal changepoints. What we "fit" is the preprocessing (standardization and 
    deseasonalization) so it's consistent between train and test data. The PELT algorithm
    itself runs fresh on each predict() call, which is the intended behavior for 
    unsupervised anomaly detection.
    """
    
    def __init__(self, penalty=10, seasonal_decomposition=False, freq=288):
        """
        Parameters
        ----------
        penalty : int, optional
            Penalty for adding changepoints. Higher values result in fewer detected changepoints.
        seasonal_decomposition : bool, optional
            Whether to deseasonalize the data before detection.
        freq : int, optional
            Frequency for seasonal decomposition (e.g., 288 for daily seasonality with 5-min intervals).
        """
        self.penalty = penalty
        self.seasonal_decomposition = seasonal_decomposition
        self.freq = freq
        self.means = {}
        self.stds = {}
        self.deseasonalisers = {}
        self.is_fitted = False
    
    def fit(self, df):
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
            
            # PELT analyzes this preprocessed data to find optimal changepoints
            # (it fits independently on each dataset, which is correct for unsupervised detection)
            algo = rpt.Pelt().fit(series.values)
            result = algo.predict(pen=self.penalty)
            
            # Create boolean anomaly series marking detected changepoints
            pred_anomalies = pd.Series(index=series.index).fillna(False)
            pred_anomalies.iloc[result[:-1]] = True
            
            anomalies[col] = pred_anomalies
        
        return anomalies
    
    def fit_predict(self, df):
        return self.fit(df).predict(df)

