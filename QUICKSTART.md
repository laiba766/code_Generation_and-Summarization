# Quick Start Guide - Phase 3

This guide will help you get started with Phase 3 of the Code Summarization and Generation project.

## Prerequisites

1. Python 3.8 or higher installed
2. pip package manager
3. (Optional) Jupyter Notebook

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import numpy, pandas, sklearn, torch; print('All dependencies installed successfully!')"
```

## Running the Project

You have two options: **Interactive (Jupyter Notebooks)** or **Automated (Main Script)**

### Option 1: Interactive Approach (Recommended for Learning)

Use Jupyter notebooks for step-by-step exploration:

```bash
# Start Jupyter
jupyter notebook notebooks/

# Then open and run in order:
# 1. 01_data_collection_and_preprocessing.ipynb
# 2. 02_clustering_experiments.ipynb
# 3. 03_prediction_experiments.ipynb
```

### Option 2: Automated Pipeline

Run the entire pipeline at once:

```bash
# Run all steps
python main.py --step all

# Or run individual steps:
python main.py --step collect --source-dir /path/to/your/code/repos
python main.py --step extract
python main.py --step cluster
python main.py --step predict
```

## Quick Example with Sample Data

If you don't have code repositories ready, here's how to test with a small sample:

### 1. Create Sample Data

Create a file `data/raw/python_functions.jsonl` with sample functions:

```json
{"function_name": "add", "code": "def add(a, b):\n    return a + b", "language": "Python", "file_path": "sample.py"}
{"function_name": "factorial", "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)", "language": "Python", "file_path": "sample.py"}
```

### 2. Extract Features

```python
import sys
sys.path.append('src')

from feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
df = extractor.process_dataset(
    'data/raw/python_functions.jsonl',
    'data/processed/python_features.csv'
)

print(df.head())
```

### 3. Run a Simple Clustering

```python
import yaml
import pandas as pd
from clustering_models import ClusteringPipeline

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

df = pd.read_csv('data/processed/python_features.csv')
pipeline = ClusteringPipeline(config)

X_scaled, features = pipeline.prepare_data(df)
results = pipeline.kmeans_clustering(X_scaled)

print(f"Silhouette Score: {results['best_score']:.4f}")
```

## Expected Outputs

After running the full pipeline, you should see:

### Directory Structure

```
Topl_project/
├── data/
│   ├── raw/
│   │   ├── python_functions.jsonl
│   │   ├── java_functions.jsonl
│   │   ├── javascript_functions.jsonl
│   │   └── rust_functions.jsonl
│   └── processed/
│       ├── python_features.csv
│       ├── java_features.csv
│       ├── javascript_features.csv
│       ├── rust_features.csv
│       ├── all_features.csv
│       └── features_with_clusters.csv
├── results/
│   ├── visualizations/
│   │   ├── tsne_plot.png
│   │   ├── pca_plot.png
│   │   ├── cluster_distribution.png
│   │   ├── elbow_curve.png
│   │   ├── feature_importance.png
│   │   └── model_comparison.png
│   ├── metrics/
│   │   ├── clustering_results.json
│   │   ├── prediction_results.json
│   │   └── model_comparison.csv
│   └── PHASE3_SUMMARY_REPORT.txt
└── models/
    └── (saved models)
```

### Sample Visualizations

1. **t-SNE Plot**: Shows how code functions cluster in 2D space
2. **Cluster Distribution**: Bar chart showing language distribution across clusters
3. **Elbow Curve**: Helps determine optimal number of clusters
4. **Feature Importance**: Which code features matter most
5. **Model Comparison**: Performance of different ML models

## Common Issues and Solutions

### Issue 1: Module Not Found

```
ModuleNotFoundError: No module named 'xyz'
```

**Solution:**
```bash
pip install xyz
# or
pip install -r requirements.txt
```

### Issue 2: CUDA/GPU Errors (LSTM Training)

```
RuntimeError: CUDA out of memory
```

**Solution:** The code automatically falls back to CPU. To force CPU:
```python
import torch
torch.device('cpu')
```

### Issue 3: Empty Dataset

```
ValueError: No data to process
```

**Solution:** Make sure you have `.jsonl` files in `data/raw/` directory.

### Issue 4: Parsing Errors

```
SyntaxError: invalid syntax in code
```

**Solution:** This is normal for some malformed code. The system skips invalid files and continues.

## Understanding the Results

### Clustering Results

**Silhouette Score** (0 to 1):
- **> 0.7**: Strong clustering
- **0.5 - 0.7**: Moderate clustering
- **< 0.5**: Weak clustering

**What to Look For:**
- Do Python and JavaScript cluster together? (Both dynamic)
- Do Java and Rust cluster together? (Both static)
- This tests your hypothesis!

### Prediction Results

**Accuracy** (0 to 1):
- **> 0.8**: Excellent - Can predict language from complexity
- **0.6 - 0.8**: Good - Some patterns detected
- **< 0.6**: Poor - Complexity alone insufficient

**What to Look For:**
- Which features are most important?
- Can we distinguish static vs dynamic languages?
- Are complexity metrics language-specific?

## Next Steps

1. **Analyze Results**: Review the visualizations in `results/visualizations/`

2. **Read the Summary**: Check `results/PHASE3_SUMMARY_REPORT.txt`

3. **Explore Notebooks**: Open Jupyter notebooks for detailed analysis

4. **Write Your Report**: Use the results to support your hypothesis

## Getting Help

### Check the Logs

The main script produces detailed logs:
```bash
python main.py --step all 2>&1 | tee pipeline.log
```

### Validate Your Data

```python
import pandas as pd

df = pd.read_csv('data/processed/all_features.csv')
print(df.info())
print(df.describe())
print(df['language'].value_counts())
```

### Test Individual Components

```python
# Test AST parser
from ast_parser import ASTAnalyzer

analyzer = ASTAnalyzer()
code = "def hello(): return 'world'"
features = analyzer.analyze(code, 'Python')
print(features)

# Test complexity metrics
from complexity_metrics import ComplexityCalculator

calc = ComplexityCalculator()
metrics = calc.calculate_all(code, 'Python')
print(metrics)
```

## Tips for Success

1. **Start Small**: Test with 100-500 functions per language first
2. **Check Data Quality**: Use the EDA notebook to validate your data
3. **Monitor Memory**: Large datasets may need PCA dimensionality reduction
4. **Save Intermediate Results**: The pipeline saves checkpoints at each step
5. **Use GPU**: For LSTM training, use a GPU if available

## Project Timeline

Estimated time for each step:

1. **Data Collection**: 30 min - 2 hours (depends on repo size)
2. **Feature Extraction**: 1-3 hours (depends on dataset size)
3. **Clustering**: 15-30 minutes
4. **Prediction**: 30 min - 2 hours (LSTM takes longer)
5. **Analysis**: 1-2 hours

**Total**: 3-8 hours for complete pipeline

## Resources

- **Phase 1 PDF**: Hypothesis and literature review
- **Phase 2 PDF**: Methodology and feature definitions
- **Config File**: `config/config.yaml` - Tune hyperparameters here
- **Source Code**: `src/` - All implementation details

## Support

For issues:
1. Check the error message
2. Review the relevant notebook
3. Validate your input data
4. Check configuration settings

---

Good luck with your experiments!

**Author**: Laiba Akram (42943)
**Course**: BSCS-7A - Theory of Programming Languages
