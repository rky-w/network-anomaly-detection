
import numpy as np
import pandas as pd

def generate_data(
        start='2026-02-01', 
        end='2026-02-02', 
        base_bytes=1000,
        peak_bytes=500,
        peak_hour=21
        ):
    
    # Get timeseries of 5 min timestamps
    ts = pd.date_range(start, end, freq='5min')
    
    # Convert to decimal values within the day
    ts_decimal = ts.hour + (ts.minute / 60) + (ts.second / 3600)

    # Create a cosine cycle to simulate patterns over the day, peaking at the specified hour
    cycle = np.cos(2 * np.pi * (ts_decimal - peak_hour) / 24)

    # Calculate bytes_in and bytes_out with some randomness
    bytes_in = np.maximum(base_bytes + (peak_bytes * cycle) + np.random.normal(0, 100, len(ts)), 0)
    bytes_out = np.maximum((bytes_in / 3) - np.random.normal(0, 50, len(ts)), 0)

    # Add random error rates with a baseline level (and a slight increase during peak hours)
    error_rate = np.minimum(np.maximum(0.05 + (0.01 * cycle) + np.random.normal(0, 0.02, len(ts)), 0), 1)

    # Create a dataframe
    df = pd.DataFrame(index = ts, data={
        'bytes_in': bytes_in,
        'bytes_out': bytes_out,
        'error_rate': error_rate
    })

    return df


def generate_point_anomalies(df, num_anomalies=10):
    """
    Randomly add spikes or drops in the metrics
    """
    df_anomalous = df.copy()
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)

    anomaly_flags = pd.DataFrame(index=df.index, columns=['bytes_in', 'bytes_out', 'error_rate'])
    
    for idx in anomaly_indices:
        # Randomly decide the type of anomaly
        anomaly_type = np.random.choice(['spike', 'drop', 'error_spike'])
        
        if anomaly_type == 'spike':
            df_anomalous.at[idx, 'bytes_in'] *= np.random.uniform(2, 3)  # Spike in bytes_in
            df_anomalous.at[idx, 'bytes_out'] *= np.random.uniform(2, 3)  # Spike in bytes_out
            anomaly_flags.at[idx, 'bytes_in'] = True
            anomaly_flags.at[idx, 'bytes_out'] = True
            
        elif anomaly_type == 'drop':
            df_anomalous.at[idx, 'bytes_in'] *= np.random.uniform(0.1, 0.2)  # Drop in bytes_in
            df_anomalous.at[idx, 'bytes_out'] *= np.random.uniform(0.1, 0.2)  # Drop in bytes_out
            anomaly_flags.at[idx, 'bytes_in'] = True
            anomaly_flags.at[idx, 'bytes_out'] = True
            
        # Note that in reality we might expect spikes and drops in traffic to correlate with spikes in error rates
        elif anomaly_type == 'error_spike':
            df_anomalous.at[idx, 'error_rate'] = min(df_anomalous.at[idx, 'error_rate'] + np.random.uniform(0.1, 0.5), 1)  # Spike in error rate
            anomaly_flags.at[idx, 'error_rate'] = True
    
    return df_anomalous, anomaly_flags.fillna(False)


def generate_level_anomalies(df, num_anomalies=1):
    """
    Randomly add sustained periods of increased traffic and error rates.
    The function ensures that level change periods do not overlap.
    Up to `num_anomalies` periods will be added, but this may be reduced if periods overlap and are stripped out.
    """
    df_anomalous = df.copy()
    
    anomaly_flags = pd.DataFrame(index=df.index, columns=['bytes_in', 'bytes_out', 'error_rate'])
    
    # Track which positions are used (including padding)
    used_positions = set()
    padding = 6
    
    for i in range(num_anomalies):    
        # Get available positions throughout the first third of the timeseries (not in used set)
        available_positions = [p for p in range(int(len(df)//3)) if p not in used_positions]
        
        # Stop if not enough positions available
        if len(available_positions) < 6:
            break
        
        # Pick random start from available positions
        start_pos = np.random.choice(available_positions)
        length = np.random.randint(48, 288*2)  # Anomalies lasting between 4 hours and 2 days
        
        # Calculate the actual range that will be marked as used
        end_pos = min(start_pos + length, len(df))
        start_remove = max(0, start_pos - padding)
        end_remove = min(end_pos + padding, len(df))
        
        # Check if this range overlaps with already used positions
        range_to_use = set(range(start_remove, end_remove))
        if range_to_use & used_positions:  # If there's any intersection, skip this anomaly
            continue
        
        level = np.random.uniform(1.2, 1.5)  # Anomaly level (e.g., 1.2x to 1.5x increase)
        
        # Get positional indices for the anomaly period
        anomaly_positions = np.arange(start_pos, end_pos)
        
        # Convert positions to actual index labels for DataFrame operations
        anomaly_indices = df.index[anomaly_positions]
        
        df_anomalous.loc[anomaly_indices, 'bytes_in'] *= level
        df_anomalous.loc[anomaly_indices, 'bytes_out'] *= level
        df_anomalous.loc[anomaly_indices, 'error_rate'] = np.minimum(df_anomalous.loc[anomaly_indices, 'error_rate'] + (level * 0.05), 1)
        
        # Mark the padded range as used
        used_positions.update(range_to_use)
        
        # Mark the start and end of the level change
        anomaly_flags.loc[df.index[start_pos], ['bytes_in', 'bytes_out', 'error_rate']] = True
        anomaly_flags.loc[df.index[np.minimum(end_pos, len(df)-1)], ['bytes_in', 'bytes_out', 'error_rate']] = True
    
    return df_anomalous, anomaly_flags.fillna(False)


def generate_trend_anomalies(df):
    """
    Create a change in the underlying trend of the data.

    """

    df_anomalous = df.copy()

    anomaly_flags = pd.DataFrame(index=df.index, columns=['bytes_in', 'bytes_out', 'error_rate'])

    # Allow trend change starts in the second half of the series
    start = np.random.choice(range(len(df)//2, int(len(df)//1.1)))
    length = len(df) - start

    anomaly_indices = df.index[start:start+length]

    m = np.random.uniform(0.01, 0.02) # Random gradient increase of 1 to 2 percentage points per time step

    for i, pos in enumerate(anomaly_indices):
        # df_anomalous.at[pos, 'bytes_in'] *= (1 + (m * i))
        # df_anomalous.at[pos, 'bytes_out'] *= (1 + (m * i))
        df_anomalous.at[pos, 'error_rate'] = np.minimum(df_anomalous.at[pos, 'error_rate'] + ((m * i) * 0.03), 1) # Scaling factor for error_rate increase due to lower base values
        if i == 0:
            anomaly_flags.loc[pos, 'error_rate'] = True

    return df_anomalous, anomaly_flags.fillna(False)

