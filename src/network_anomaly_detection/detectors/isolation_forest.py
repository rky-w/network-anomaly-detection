
import numpy as np
import pandas as pd
from adtk.transformer import ClassicSeasonalDecomposition
from pyod.models.iforest import IForest

class IForestAD:
    def __init__(self, contamination=0.01, seasonal_decomposition=False, freq=288):
        self.contamination = contamination
        self.seasonal_decomposition = seasonal_decomposition
        self.freq = freq
        self.means = {}
        self.stds = {}
        self.deseasonalisers = {}
        self.model = IForest(contamination=self.contamination)
        self.is_fitted = False


    def engineer_features(self, df):
        """
        Engineer features to help Isolation Forest detect anomalies.
        
        Features include:
        - Rolling statistics (mean, std, min, max)
        - Ratios and interactions between features
        - Deviations from rolling averages
        - Temporal features
        """
        df_features = df.copy()
        
        # 1. Rolling statistics for each metric (window = 12 means 1 hour with 5-min intervals)
        for col in ['bytes_in', 'bytes_out', 'error_rate']:
            df_features[f'{col}_rolling_mean_12'] = df[col].rolling(window=12, min_periods=1).mean()
            df_features[f'{col}_rolling_std_12'] = df[col].rolling(window=12, min_periods=1).std().fillna(0)
            df_features[f'{col}_rolling_min_12'] = df[col].rolling(window=12, min_periods=1).min()
            df_features[f'{col}_rolling_max_12'] = df[col].rolling(window=12, min_periods=1).max()
            
            # Deviation from rolling mean (z-score like)
            std = df_features[f'{col}_rolling_std_12'].replace(0, 1)  # Avoid division by zero
            df_features[f'{col}_deviation'] = (df[col] - df_features[f'{col}_rolling_mean_12']) / std
        
        # 2. Longer window rolling statistics (window = 48 means 4 hours)
        for col in ['bytes_in', 'bytes_out', 'error_rate']:
            df_features[f'{col}_rolling_mean_48'] = df[col].rolling(window=48, min_periods=1).mean()
            df_features[f'{col}_rolling_std_48'] = df[col].rolling(window=48, min_periods=1).std().fillna(0)
        
        # 3. Feature interactions and ratios
        df_features['bytes_ratio'] = df['bytes_out'] / (df['bytes_in'] + 1)  # Avoid division by zero
        df_features['total_bytes'] = df['bytes_in'] + df['bytes_out']
        df_features['bytes_diff'] = df['bytes_out'] - df['bytes_in']
        
        # Error rate weighted by traffic volume
        df_features['error_weighted'] = df['error_rate'] * (df_features['bytes_in'] + df_features['bytes_out'])
        
        # 4. Rate of change (first derivative)
        for col in ['bytes_in', 'bytes_out', 'error_rate']:
            df_features[f'{col}_rate_of_change'] = df[col].diff().fillna(0)
            
        # 5. Temporal features
        df_features['hour'] = df.index.hour
        df_features['day_of_week'] = df.index.dayofweek
        df_features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Cyclic encoding of hour (to capture periodicity)
        df_features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # 6. Moving range (volatility measure)
        for col in ['bytes_in', 'bytes_out', 'error_rate']:
            df_features[f'{col}_range_12'] = (
                df_features[f'{col}_rolling_max_12'] - df_features[f'{col}_rolling_min_12']
            )
        
        return df_features
    
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

        df = df.copy()

        for col in df.columns:
            series = df[col]
            
            # Apply standardization using stored parameters from fit()
            series = (series - self.means[col]) / self.stds[col]
            
            # Apply deseasonalization if fitted
            if self.seasonal_decomposition:
                series = self.deseasonalisers[col].transform(series)

            df[col] = series

        self.model.fit(self.engineer_features(df))
        anomalies = self.model.predict(self.engineer_features(df))

        return pd.Series(anomalies, index=df.index).astype(bool)
    
    def fit_predict(self, df):
        self.fit(df)
        return self.predict(df)
