# CERA CLI

Context-Engineered Reviews Architecture - Python CLI for synthetic ABSA dataset generation.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Run with config file
cera config.json

# With options
cera config.json --output restaurant-reviews-2000
cera config.json -n 500 --noise-preset moderate

# Validate config
cera config.json --dry-run

# Start API server
cera serve

# Evaluate generated dataset
cera evaluate dataset.jsonl --reference real_reviews.jsonl

# Export to different formats
cera export dataset.jsonl --format csv
```
