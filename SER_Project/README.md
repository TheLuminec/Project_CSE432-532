# SER Project - Speech Emotion Recognition with MiniLearn

Speech Emotion Recognition on the RAVDESS audio-only dataset using a custom
`minilearn` package for from-scratch machine learning models and utilities.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Linux/Mac
source ./venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the RAVDESS audio-only data
python download_data.py

# 5. Extract metadata + audio features into features.csv
python parser.py

# 6. Open the main SER notebook
jupyter notebook notebooks/RAVDESS.ipynb
```

## Project Structure

```text
SER_Project/
|-- minilearn/                # From-scratch ML library
|   |-- models/              # LR, KNN, NB, SVM, tree, clustering, ANN
|   |-- preprocessing.py     # StandardScaler, train_test_split
|   |-- metrics.py           # Accuracy, precision, recall, F1, ROC/AUC, confusion matrix
|   |-- model_selection.py   # KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
|   |-- dim_reduction.py     # PCA
|   |-- ensemble.py          # Voting ensemble helper
|   `-- classifiers.py       # Convenience exports
|-- notebooks/
|   |-- RAVDESS.ipynb        # Main SER experiment notebook
|   |-- eda.ipynb            # Early data exploration
|   |-- minilearn.ipynb      # MiniLearn experiments/comparisons
|   `-- cancer_tests.ipynb   # Sanity checks on standard datasets
|-- parser.py                # WAV parsing + handcrafted feature extraction
|-- download_data.py         # Dataset download helper
|-- requirements.txt         # Python dependencies
`-- README.md
```

## Dataset

This project uses the RAVDESS audio-only release from Zenodo:
https://zenodo.org/records/1188976

The target label is the `emotion` field parsed from the RAVDESS filename.

## MiniLearn

Example usage:

```python
from minilearn.classifiers import LogisticRegression, KNN, GaussianNaiveBayes
from minilearn.preprocessing import StandardScaler, train_test_split
from minilearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
```
