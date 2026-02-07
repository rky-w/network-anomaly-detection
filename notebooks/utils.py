import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ruptures as rpt
from plotly.subplots import make_subplots
from adtk.transformer import ClassicSeasonalDecomposition

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
def plot_anomalies(df, anomalies, title='Anomalies in Network Traffic'):
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


# Pelt anomaly detection
def _pelt_ad(series, penalty=10, seasonal_decomposition=False, plotting=False):


    m = np.mean(series)
    s = np.std(series)
    series = (series - m) / s  # Standardize the series

    if seasonal_decomposition:
        deseasonaliser = ClassicSeasonalDecomposition(freq=288)
        series = deseasonaliser.fit_transform(series)
    
    # detection
    algo = rpt.Pelt().fit(series.values)
    result = algo.predict(pen=penalty)

    # display
    if plotting:
        rpt.display(series, result)
        plt.show()

    # Create output series with True for detected change points
    anomalies = pd.Series(index=series.index).fillna(False)
    anomalies.iloc[result[:-1]] = True

    return anomalies


def pelt_ad(df, penalty=10, seasonal_decomposition=False):
    """
    Use of the PELT algorithm for change point detection. 
    The penalty parameter controls the sensitivity of the algorithm to detecting change points; higher values will result in fewer detected change points, while lower values will allow for more detections.
    """

    anomalies = pd.DataFrame(index=df.index)

    for col in df.columns:
        anomalies[col] = _pelt_ad(df[col], penalty=penalty, seasonal_decomposition=seasonal_decomposition)
        
    return anomalies





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


