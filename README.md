# Code Summarization and Generation Across Programming Languages

**Phase 3: Model Implementation & Experimentation**

## ✨ NEW: AI-Powered Code Generation & Summarization

🚀 **Write in English, Get Code** - Describe what you want, get working code  
🤖 **Paste Code, Get Summary** - Understand any code in plain English

👉 **Quick Start:** `python demo_quick.py`  
📚 **Full Guide:** See [QUICKSTART_CODE_GEN.md](QUICKSTART_CODE_GEN.md)

---

## Project Overview

This project investigates how programming language features (typing discipline, paradigm, memory safety) and syntax complexity metrics influence LLM performance in code generation and summarization.

### Author
- **Name:** Laiba Akram
- **Student ID:** 42943
- **Course:** BSCS-7A - Theory of Programming Languages

### Research Question
How do programming language features (typing discipline, paradigm, memory safety) and syntax complexity metrics (AST-based and traditional) influence LLM performance in code generation and code summarization?

### Hypothesis
For function-level tasks of equivalent functionality, LLM-generated code in statically typed, memory-safe languages (e.g., Rust, Go, Java) will exhibit higher syntax complexity (e.g., deeper ASTs, higher cyclomatic complexity) than generated code in dynamically typed languages (e.g., Python, JavaScript), and these complexity differences will correlate with summarization quality across languages.

## Project Structure

```
Topl_project/
├── config/
│   └── config.yaml              # Configuration parameters
├── data/
│   ├── raw/                     # Raw collected code snippets
│   └── processed/               # Processed features and datasets
├── src/
│   ├── __init__.py
│   ├── data_collector.py        # Code collection from repositories
│   ├── ast_parser.py            # AST analysis and feature extraction
│   ├── complexity_metrics.py    # McCabe and Halstead metrics
│   ├── feature_extractor.py     # Combined feature extraction pipeline
│   ├── clustering_models.py     # K-Means, DBSCAN, Hierarchical clustering
│   ├── prediction_models.py     # Random Forest, LSTM, Transformer models
│   ├── code_generator.py        # 🆕 AI code generation from English
│   ├── code_summarizer.py       # 🆕 AI code summarization to English
│   └── visualization.py         # Visualization utilities
├── notebooks/
│   ├── 01_data_collection_and_preprocessing.ipynb
│   ├── 02_clustering_experiments.ipynb
│   └── 03_prediction_experiments.ipynb
├── models/                      # Saved trained models
├── results/
│   ├── visualizations/          # Generated plots and figures
│   └── metrics/                 # Evaluation metrics and results
├── code_interface.py            # 🆕 Interactive code gen/summarization tool
├── demo_quick.py                # 🆕 Quick demo of new features
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Features

### 🆕 AI-Powered Features
- **Code Generation:** Write English descriptions → Get working code
- **Code Summarization:** Paste code → Get natural language explanation
- **Multi-language Support:** Python, Java, JavaScript
- **Batch Processing:** Analyze multiple files at once
- **Detailed Analysis:** Extract complexity, dependencies, operations

### Programming Language Features Analyzed
- **Typing Discipline:** Static vs Dynamic
- **Memory Model:** Manual, Managed, Ownership
- **Paradigm:** Imperative, OOP, Functional
- **Concurrency Model:** Threading, Async/Await, Event Loop

### Syntax Complexity Metrics
- **AST-based Metrics:**
  - Node count
  - Tree depth
  - Branching factor
  - Distinct node types
  - Leaf count

- **Traditional Metrics:**
  - McCabe Cyclomatic Complexity
  - Halstead Metrics (Volume, Difficulty, Effort)

- **Control Flow Features:**
  - Number of conditionals, loops, exception handling
  - Function calls
  - Statement and expression counts

### Concurrency Features
- Async/await patterns
- Thread creation and management
- Mutex/lock usage
- Promise/Future patterns
- Channel/message passing

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Data Collection

Collect code snippets from your repositories:

```python
from src.data_collector import DataCollector
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

collector = DataCollector(config)
functions = collector.collect_from_directory('/path/to/repositories')
collector.save_dataset(functions, 'data/raw')
```

### 2. Feature Extraction

Extract AST and complexity features:

```python
from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# Process each language
for language in ['python', 'java', 'javascript', 'rust']:
    input_file = f'data/raw/{language}_functions.jsonl'
    output_file = f'data/processed/{language}_features.csv'
    df = extractor.process_dataset(input_file, output_file)
```

### 3. Run Clustering Experiments

```python
from src.clustering_models import ClusteringPipeline
import pandas as pd

df = pd.read_csv('data/processed/all_features.csv')
pipeline = ClusteringPipeline(config)

X_scaled, feature_df = pipeline.prepare_data(df)
X_pca = pipeline.apply_pca(X_scaled)
results = pipeline.run_all_clustering(X_pca)
```

### 4. Run Prediction Experiments

```python
from src.prediction_models import RandomForestPipeline

rf_pipeline = RandomForestPipeline(config, task='classification')
X_train, X_test, y_train, y_test = rf_pipeline.prepare_data(df, 'language')

train_results = rf_pipeline.train(X_train, y_train)
test_results = rf_pipeline.evaluate(X_test, y_test)
```

### 5. Use Jupyter Notebooks

For interactive experimentation, use the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

## Models Implemented

### Clustering Models
1. **K-Means:** Partitioning-based clustering with elbow method
2. **DBSCAN:** Density-based clustering for arbitrary shapes
3. **Hierarchical Clustering:** Agglomerative clustering with different linkage methods

### Prediction Models
1. **Random Forest:** Tree-based ensemble for classification
2. **LSTM:** Recurrent neural network for sequence modeling
3. **Transformer (CodeBERT):** Pre-trained code understanding model

## Evaluation Metrics

### Clustering Metrics
- Silhouette Score
- Davies-Bouldin Score
- Calinski-Harabasz Score

### Prediction Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Visualizations

The project generates various visualizations:
- **t-SNE plots:** 2D visualization of high-dimensional data
- **PCA plots:** Principal component analysis
- **Cluster distribution:** Language distribution across clusters
- **Elbow curves:** Optimal number of clusters
- **Feature importance:** Most influential features
- **Confusion matrices:** Classification performance
- **Metric distributions:** Box plots of complexity metrics

## Results

Results are saved in the `results/` directory:
- `results/visualizations/`: All generated plots
- `results/metrics/`: JSON files with evaluation metrics

## Key Findings

*To be completed after running experiments:*

1. **Clustering Analysis:**
   - Do languages cluster by typing discipline?
   - Do memory-safe languages exhibit distinct patterns?

2. **Prediction Accuracy:**
   - Can we predict the programming language from complexity metrics?
   - Which features are most important?

3. **Hypothesis Validation:**
   - Do statically-typed languages have higher syntax complexity?
   - How does this correlate with code summarization quality?

## Configuration

Edit `config/config.yaml` to customize:
- Data collection parameters
- Sampling strategy
- Model hyperparameters
- Visualization settings

## Dependencies

Core libraries:
- numpy, pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- torch: Deep learning (LSTM)
- transformers: CodeBERT models
- matplotlib, seaborn, plotly: Visualization
- radon, lizard: Code complexity analysis

## Future Work

- Expand to more programming languages (Go, C++, Swift)
- Implement code summarization quality metrics
- Fine-tune transformer models on code data
- Cross-language code generation experiments

## References

This project builds upon:
- Phase 01: Literature review and hypothesis formulation
- Phase 02: Methodology and data preprocessing

## License

This project is for academic purposes only.

## Contact

For questions or collaboration:
- **Name:** Laiba Akram
- **Student ID:** 42943
- **Course:** BSCS-7A Theory of Programming Languages

---

**Date:** December 2024
