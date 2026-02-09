import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
## Evaluation
###########################################

# def anomaly_classification_report(actual, predicted):

#     metrics = ['bytes_in', 'bytes_out', 'error_rate']

#     for i, metric in enumerate(metrics):
#         print(f"Classification Report for {metric} Anomalies:")
#         print(classification_report(actual.iloc[:, i], predicted.iloc[:, i], target_names=['Normal', 'Anomaly']))
#         cm = confusion_matrix(actual.iloc[:, i], predicted.iloc[:, i])
#         print(f"True Negatives: {cm[0,0]:>6} | False Positives: {cm[0,1]:>6}")
#         print(f"False Negatives: {cm[1,0]:>5} | True Positives:  {cm[1,1]:>6}")
#         print("-"*75)
#         print("")

def anomaly_classification_report(actual, predicted, target_names=('Normal', 'Anomaly')):
    """
    Print classification reports and confusion matrices for anomaly detection.

    Parameters
    ----------
    actual : pd.Series or pd.DataFrame
        Ground truth labels (0/1 or False/True).
    predicted : pd.Series or pd.DataFrame
        Predicted labels (0/1 or False/True).
    target_names : tuple[str, str]
        Names for the negative and positive classes.
    """

    # --- Normalise inputs ---
    if isinstance(actual, pd.Series):
        actual = actual.to_frame(name=actual.name or "overall")

    if isinstance(predicted, pd.Series):
        predicted = predicted.to_frame(name=predicted.name or "overall")

    # --- Basic validation ---
    if actual.shape != predicted.shape:
        raise ValueError("`actual` and `predicted` must have the same shape")

    if not actual.columns.equals(predicted.columns):
        raise ValueError("`actual` and `predicted` must have the same columns")

    # --- Generate reports ---
    for col in actual.columns:
        print(f"Classification Report for '{col}' anomalies:")
        print(
            classification_report(
                actual[col],
                predicted[col],
                target_names=target_names,
                zero_division=0,
            )
        )

        cm = confusion_matrix(actual[col], predicted[col])
        tn, fp, fn, tp = cm.ravel()

        print(f"True Negatives: {tn:>6} | False Positives: {fp:>6}")
        print(f"False Negatives: {fn:>5} | True Positives:  {tp:>6}")
        print("-" * 75)
        print()




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


