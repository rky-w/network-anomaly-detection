# AWS Deployment Guide

Minimal serverless deployment for network anomaly detection using Lambda, DynamoDB, and EventBridge.

## Prerequisites

- AWS CLI configured with appropriate credentials
- AWS SAM CLI installed
- Python 3.11+

## Deployment Steps

1. **Install dependencies locally** (for Lambda layer):
   ```bash
   cd deployment
   pip install -r requirements.txt -t ./package
   ```

2. **Deploy with SAM**:
   ```bash
   sam build
   sam deploy --guided
   ```

   On first deployment, you'll be prompted for:
   - Stack name (e.g., `network-anomaly-detection`)
   - AWS Region (e.g., `us-east-1`)
   - Confirm IAM role creation

3. **Verify deployment**:
   ```bash
   aws lambda list-functions | grep NetworkAnomalyDetector
   aws dynamodb list-tables | grep NetworkMetrics
   ```

## Testing

Insert sample metrics into DynamoDB:

```python
import boto3
from datetime import datetime, timedelta

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('NetworkMetrics')

# Insert sample data point
table.put_item(
    Item={
        'timestamp': datetime.now().isoformat(),
        'bytes_in': 1000000,
        'bytes_out': 500000,
        'error_rate': 0.01,
        'ttl': int((datetime.now() + timedelta(days=7)).timestamp())
    }
)
```

Invoke Lambda manually to test:

```bash
aws lambda invoke \
    --function-name NetworkAnomalyDetector \
    --payload '{}' \
    response.json
cat response.json
```

## Monitoring

- **Lambda logs**: Check CloudWatch Logs for execution details
- **Detected anomalies**: Query the `NetworkAnomalies` DynamoDB table
- **Metrics**: Monitor Lambda duration, error rates in CloudWatch

## Cost Estimate

For typical usage (3 metrics, 5-minute intervals):
- Lambda: ~$5-10/month
- DynamoDB: ~$5-15/month (depends on data volume)
- EventBridge: Minimal (<$1/month)

**Total**: ~$10-25/month

## Cleanup

Remove all resources:

```bash
sam delete --stack-name network-anomaly-detection
```
