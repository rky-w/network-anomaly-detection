import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ruptures as rpt
from plotly.subplots import make_subplots
from adtk.transformer import ClassicSeasonalDecomposition
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


###########################################
## Plotting
###########################################

# Plot the clean network traffic data
def plot_network_metrics(df, title='Network Traffic Metrics'):

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.bytes_in,
        name='Bytes In'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.bytes_out,
        name='Bytes Out'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.error_rate,
        name='Error Rate',
    ), row=2, col=1)

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))

    fig.update_layout(height=600, width=1000, title_text=title)
    fig.show()


# Plot anomalies for the metrics in the network traffic data
def plot_anomalies(df, anomalies, actuals=None, title='Anomalies in Network Traffic'):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    metrics = ['bytes_in', 'bytes_out', 'error_rate']

    # Ensure no missing values in anomaly dfs
    anomalies = anomalies.fillna(False)

    for i, metric in enumerate(metrics, start=1):
            
            
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[metric],
            name=metric
        ), row=i, col=1)

        # Add anomalies as red markers
        fig.add_trace(go.Scatter(
            x=df[anomalies[metric]].index,
            y=df[anomalies[metric]][metric],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomaly',
            showlegend=False
        ), row=i, col=1)

        if actuals is not None:
            anomaly_times = df.index[actuals[metric]]
            for anomaly_time in anomaly_times:
                fig.add_vline(
                    x=anomaly_time,
                    line=dict(color='green', width=1),
                    opacity=0.7,
                    row=i, col=1
                )

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))

    fig.update_layout(height=600, width=1000, title_text=title)
    fig.show()

# Plot multivariate anomalies for the metrics in the network traffic data
def plot_multivariate_anomalies(df, anomalies, title='Anomalies in Network Traffic'):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    metrics = ['bytes_in', 'bytes_out', 'error_rate']

    # Ensure no missing values in anomaly dfs
    anomalies = anomalies.fillna(False)

    for i, metric in enumerate(metrics, start=1):
            
            
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[metric],
            name=metric
        ), row=i, col=1)

        # Add anomalies as vertical purple lines

        anomaly_times = df.index[anomalies]
        for anomaly_time in anomaly_times:
            fig.add_vline(
                x=anomaly_time,
                line=dict(color='purple', width=1),
                opacity=0.7,
                row=i, col=1
            )

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))

    fig.update_layout(height=600, width=1000, title_text=title)
    fig.show()



###########################################
## Detection
###########################################

# CUSUM-based anomaly detection
def _cusum_ad(series, threshold=5, seasonal_decomposition=False):
    """
    Simple CUSUM-based trend change detector.
    For each sequential group of anomaly flags, only the first index is marked as True.
    """

    m = np.mean(series)
    s = np.std(series)
    series = (series - m) / s  # Standardize the series

    if seasonal_decomposition:
        deseasonaliser = ClassicSeasonalDecomposition(freq=288)
        series = deseasonaliser.fit_transform(series)

    cumsum = np.cumsum(series - series.mean())
    anomalies = np.abs(cumsum) > threshold
    
    # Need to keep only the first index of each sequential group of anomalies
    anomalies_series = pd.Series(anomalies, index=series.index)
    changes = anomalies_series.astype(int).diff().fillna(0)
    # Transitions from False to True (value of 1) indicate the start of anomaly groups
    result = (changes == 1)
    
    return result

def cusum_ad(df, threshold=250, seasonal_decomposition=False):
    """
    Apply CUSUM-based anomaly detection to each column in the dataframe.
    """
    anomalies = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        anomalies[col] = _cusum_ad(df[col], threshold, seasonal_decomposition=seasonal_decomposition)
    
    return anomalies


# PELT anomaly detection with fit/predict interface
class PELTADDetector:
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


# Legacy function-based interface (for backwards compatibility)
def pelt_ad(df, penalty=10, seasonal_decomposition=False):
    """
    Use of the PELT algorithm for change point detection. 
    The penalty parameter controls the sensitivity of the algorithm to detecting change points; higher values will result in fewer detected change points, while lower values will allow for more detections.
    
    Note: This is a convenience function. For train/test workflows, use PELTADDetector class instead.
    """
    detector = PELTADDetector(penalty=penalty, seasonal_decomposition=seasonal_decomposition)
    return detector.fit_predict(df)





###########################################
## Evaluation
###########################################

def anomaly_classification_report(actual, predicted):

    metrics = ['bytes_in', 'bytes_out', 'error_rate']

    for i, metric in enumerate(metrics):
        print(f"Classification Report for {metric} Anomalies:")
        print(classification_report(actual.iloc[:, i], predicted.iloc[:, i]))
        cm = confusion_matrix(actual.iloc[:, i], predicted.iloc[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
        disp.plot(ax=plt.gca())
        plt.title(f'Confusion Matrix for {metric}')
        plt.show()
        print("-"*75)
        print("")





###########################################
## General
###########################################


def combine_anomalies(dataframes):
    """
    OR logic to combine anomaly detection dataframes
    """
    result = dataframes[0].copy()

    for df in dataframes[1:]:
        result = result | df
        
    return result


