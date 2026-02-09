"""
AWS Lambda handler for network anomaly detection
Runs ensemble pipeline on rolling window of metrics from DynamoDB
"""
import json
import os
from datetime import datetime, timedelta
import boto3
import pandas as pd
from network_anomaly_detection.pipeline.ensemble import run_ensemble_pipeline

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])


def lambda_handler(event, context):
    """
    Lambda handler triggered every 5 minutes to detect anomalies
    in network metrics stored in DynamoDB
    """
    
    # Define rolling window (last 24 hours)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    # Query DynamoDB for recent metrics
    metrics_data = query_metrics(start_time, end_time)
    
    if not metrics_data:
        return {
            'statusCode': 200,
            'body': json.dumps('No data found in window')
        }
    
    # Convert to pandas DataFrame with timestamp index
    df = pd.DataFrame(metrics_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Run ensemble anomaly detection on each metric
    results = {}
    for metric in ['bytes_in', 'bytes_out', 'error_rate']:
        if metric in df.columns:
            anomalies = run_ensemble_pipeline(df[[metric]])
            results[metric] = anomalies.sum()  # Count of anomalies detected
            
            # Store anomalies back to DynamoDB
            store_anomalies(metric, anomalies)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Anomaly detection completed',
            'window': f'{start_time} to {end_time}',
            'anomalies_detected': results
        })
    }


def query_metrics(start_time, end_time):
    """Query DynamoDB for metrics in time window"""
    
    response = table.query(
        KeyConditionExpression='#ts BETWEEN :start AND :end',
        ExpressionAttributeNames={'#ts': 'timestamp'},
        ExpressionAttributeValues={
            ':start': start_time.isoformat(),
            ':end': end_time.isoformat()
        }
    )
    
    return response.get('Items', [])


def store_anomalies(metric_name, anomalies):
    """Store detected anomalies to DynamoDB anomalies table"""
    
    anomalies_table = dynamodb.Table(os.environ['ANOMALIES_TABLE'])
    
    # Only store timestamps where anomalies were detected
    anomaly_timestamps = anomalies[anomalies].index
    
    for ts in anomaly_timestamps:
        anomalies_table.put_item(
            Item={
                'metric': metric_name,
                'timestamp': ts.isoformat(),
                'detected_at': datetime.now().isoformat()
            }
        )
