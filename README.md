# Network Anomaly Detection

An example project demonstrating anomaly detection in network traffic using unsupervised machine learning algorithms.

## Overview

This project applies unsupervised learning techniques to identify abnormal patterns in network traffic data. It explores various anomaly detection algorithms and their effectiveness on network datasets.

## Project Structure

- **notebook/** - Main exploration and analysis of anomaly detection techniques
- **src/** - Core logic and implementation of detection algorithms
- **tests/** - Pytest test suite for validating functionality
- **deployment/** - Basic AWS deployment configuration

## Requirements

- Python 3.13+
- Poetry
- Dependencies listed in `pyproject.toml`

## Getting Started

**Install dependencies locally**:
```bash
# Run setup script
source setup.sh
```
or 
```bash
# Install dependencies
poetry install

# Create kernel for notebooks
poetry run ipython kernel install --name "network-anomaly-detection" --user
```

**Run the notebook** to explore the anomaly detection analysis and methodology.

**Run tests** to validate the implementation:
```bash
pytest tests/
```

## Deployment

See the `deployment/` directory for basic AWS deployment configuration (currently untested).


