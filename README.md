# CSE 432/532 — Machine Learning Semester Project

## Speech Emotion Recognition (SER) Using the RAVDESS Dataset

> **Repository Setup:** Add the instructor as a collaborator to your repository — **@sahamis** (available on both GitLab and GitHub).

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. The Dataset — RAVDESS (Audio-Only)](#2-the-dataset--ravdess-audio-only)
- [3. The MiniLearn Library](#3-the-minilearn-library)
- [4. Feature Extraction from Audio](#4-feature-extraction-from-audio)
- [5. ML Methods & Evaluation](#5-ml-methods--evaluation)
- [6. Weekly Roadmap](#6-weekly-roadmap)
- [7. Setup](#7-setup)
- [8. Final Submission](#8-final-submission)
- [9. Rubric](#9-rubric)
- [10. Academic Integrity](#10-academic-integrity)
- [References](#references)

---

## 1. Project Overview

In this semester-long project, you will build an end-to-end **Speech Emotion Recognition (SER)** system. Starting from raw audio recordings of emotional speech, you will extract meaningful features, apply a range of machine learning algorithms to classify the expressed emotion, and write a technical report comparing your methods and results.

A unique component of this project is that you will develop your own lightweight ML toolkit — a Python package called **MiniLearn** — that implements core algorithms from scratch. You will use this package alongside scikit-learn in your SER experiments.

> **Important:** We are working exclusively with the **audio** portion of the RAVDESS dataset. You will **not** use any video or facial expression data. All your work should focus on extracting information from the audio signal only.

### Learning Objectives

By completing this project you will be able to:

- Work with a real-world audio dataset from download through analysis
- Extract and engineer features from audio signals
- Implement core ML algorithms from scratch inside a reusable library
- Apply and compare classical and modern classification techniques
- Perform model evaluation, cross-validation, and hyperparameter tuning
- Explore unsupervised learning (clustering) applied to the same problem
- Write a technical report with critical discussion of results
- Demonstrate deep understanding of every piece of code you submit

### Development Practices

One of the goals of this project is to give you hands-on experience with workflows that mirror real-world software development. This goes beyond ML — it includes how you manage your code and collaborate.

- **Stay in sync with the upstream repository.** Parts of this README and project materials may be updated during the semester. Make it a habit to regularly run `git pull` on your local repository so you always have the latest information.
- **Use proper, atomic commits.** Each commit should capture a single logical change — implementing a new feature, fixing a bug, adding a visualization, etc. Do not treat Git as a backup tool where you dump large snapshots of your entire project at once. A clear, incremental commit history shows how your work evolved and demonstrates genuine understanding.
- **Write meaningful commit messages.** A good commit message briefly describes _what_ changed and _why_. This is a professional skill that pays dividends in any collaborative environment.

We do not formally teach Git in this course, but bringing together the right tools and practices is part of what makes this a real project rather than a textbook exercise. Your commit history is part of how your work is evaluated.

---

## 2. The Dataset — RAVDESS (Audio-Only)

**The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**

- **Source:** [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976)
- **Citation (required in your report):** Livingstone SR, Russo FA (2018). _The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English._ PLoS ONE 13(5): e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)

### What to Download

Download **only** the two **audio-only** zip files from the Zenodo page:

| File                            | Size    | Contents                                      |
| ------------------------------- | ------- | --------------------------------------------- |
| `Audio_Speech_Actors_01-24.zip` | ~215 MB | 1,440 speech files (60 per actor × 24 actors) |
| `Audio_Song_Actors_01-24.zip`   | ~198 MB | 1,012 song files (44 per actor × 23 actors\*) |

_\*Actor 18 has no song files._

All audio files are **16-bit, 48 kHz WAV** format. Do **not** download the video files.

### File Naming Convention

Each filename is a 7-part numerical identifier. For example: `03-01-05-01-02-01-12.wav`

| Position | Meaning       | Values                                                                                                |
| -------- | ------------- | ----------------------------------------------------------------------------------------------------- |
| 1        | Modality      | 01 = full-AV, 02 = video-only, **03 = audio-only**                                                    |
| 2        | Vocal channel | **01 = speech**, **02 = song**                                                                        |
| 3        | **Emotion**   | 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised |
| 4        | Intensity     | 01 = normal, 02 = strong (no strong for neutral)                                                      |
| 5        | Statement     | 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"                              |
| 6        | Repetition    | 01 = 1st, 02 = 2nd                                                                                    |
| 7        | Actor         | 01–24 (odd = male, even = female)                                                                     |

Your classification target is the **Emotion** (position 3). You must write code to parse these filenames and build a metadata table.

---

## 3. The MiniLearn Library

You will develop **MiniLearn** — your own mini scikit-learn–style Python package, implemented from scratch.

### Requirements

1. **Importable Python package (miniLearn):** You should be able to write statements like `from minilearn.classifiers import LogisticRegression` in your SER notebook. MiniLearn is a mini replacement for scikit-learn, built from scratch.
   1. **[Recommended]** Follow the scikit-learn API pattern. Each model should have `.fit(X, y)`, `.predict(X)`, and `.score(X, y)` methods, consistent with how scikit-learn models work.
   2. **Required implementations** (from scratch, using only NumPy/SciPy for numerical operations):
      - Preprocessing: feature standardization, train-test split
      - Logistic Regression (with gradient descent)
        > **Tip:** For the learning/optimization part, you can use optimizers like `scipy.optimize.minimize` or similar, but you need to write the rest of the function yourself (e.g., logistic regression with softmax).
      - k-Nearest Neighbors (KNN)
      - Gaussian Naive Bayes
      - SVM (simplified linear)
      - Decision Tree (CART algorithm)
      - Evaluation metrics: accuracy, precision, recall, F1 score, confusion matrix
      - Cross-validation utility (k-fold)
      - Clustering (see Section 5)
      - PCA for dimensionality reduction
      - ANN (your choice, a 1-layer perceptron based ANN for classification is the minimum requirement)
   3. **Optional / Bonus:**
      - You can implement other methods as well to improve your comparative analysis
2. **Use MiniLearn to analyze SER classification results.** For each supervised model you implement, you will apply it to the SER data and compute all required metrics (accuracy, precision, recall, F1, confusion matrix, ROC/AUC). You will write a detailed analysis of the results — which emotions are classified well vs. poorly, how does performance compare with scikit-learn, what do the confusion matrices reveal about common misclassifications, etc. The weekly roadmap in Section 6 provides more details on which algorithms to implement and analyze each week.

> **Code Ownership:** There is an assessment for your project. You must be able to explain every function in your MiniLearn code. I reserve the right to ask you to walk through your code, explain design decisions, or modify it live during a scheduled walk-through. You should also work on the project incrementally, with a clear Git history showing the development of your MiniLearn library over time. Submitting code you cannot explain will be treated as an academic integrity violation.

---

## 4. Feature Extraction from Audio

Since this is a machine learning course (not a speech processing course), we provide guidance on audio feature extraction. Your job is to understand _what_ these features capture and _why_ they matter for emotion recognition, then to implement the extraction pipeline.

### 4.1 Hand-Crafted Features (Required)

You may use the `librosa` Python library. For each audio file, extract frame-level features and then compute **summary statistics** (mean, standard deviation, etc.) to produce a fixed-length feature vector.

| Feature                                         | Description                                                                 | Why It Matters for SER                                                       |
| ----------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **MFCCs** (Mel-Frequency Cepstral Coefficients) | Compact representation of the spectral envelope (typically 13 coefficients) | Gold standard in speech/audio ML; captures vocal tract characteristics       |
| **MFCC Deltas** (1st and 2nd order)             | Rate of change of MFCCs over time                                           | Captures dynamic aspects of speech — how the sound _changes_                 |
| **Chroma Features**                             | 12-dimensional pitch class profile                                          | Captures tonal content; useful for distinguishing emotions in song           |
| **Mel Spectrogram**                             | Time-frequency representation on the Mel scale                              | Rich spectral information; good input for CNN-based approaches               |
| **Zero Crossing Rate (ZCR)**                    | Rate at which signal changes sign                                           | Indicates noisiness; excited emotions (angry, happy) tend to have higher ZCR |
| **RMS Energy**                                  | Root mean square energy per frame                                           | Loudness — directly related to emotional intensity                           |
| **Spectral Centroid**                           | "Center of mass" of the spectrum                                            | Brightness of sound; higher for excited emotions                             |
| **Spectral Bandwidth**                          | Width of the spectral band                                                  | Spread of frequencies; varies with emotion                                   |
| **Spectral Rolloff**                            | Frequency below which 85% of energy lies                                    | Distinguishes harmonic vs. noisy signals                                     |

**How to use `librosa`:** Each feature is computed per-frame (e.g., `librosa.feature.mfcc()` returns a matrix of shape `[n_mfcc, n_frames]`). Since each audio file has different length, you should compute **summary statistics** over the frames (mean, std, min, max, etc.) to get a single fixed-length vector per file.

Example for one feature:

```python
import librosa
import numpy as np

y, sr = librosa.load("path/to/file.wav", sr=48000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)   # shape: (13, n_frames)
mfcc_mean = np.mean(mfcc, axis=1)                       # shape: (13,) — one mean per coefficient
mfcc_std = np.std(mfcc, axis=1)                         # shape: (13,)
```

Often using only one of these hand-crafted feature sets (e.g. MFCCs) is not enough to achieve good performance. You may want to experiment with different combinations of features (e.g. MFCCs and their deltas, chroma, ZCR, etc.) to see which combination works best for your classifiers. Combining multiple features is as simple as concatenating them together into one long feature vector per audio file.

### 4.2 Pre-Trained Embeddings (To boost the accuracy)

For more advanced experiments, you may extract embeddings from pre-trained audio neural networks (e.g., OpenL3, VGGish, wav2vec 2.0, or HuBERT via HuggingFace, etc.). These produce high-dimensional vectors that encode rich acoustic information. Compare the results with your hand-crafted features.

### 4.3 Feature Standardization

**Standardize your features before classification.** Audio features live on very different scales (e.g., MFCCs vs. spectral centroid in Hz vs. RMS energy). Apply proper standardization as we covered in class (e.g., z-score normalization) as appropriate for your features. One particular quality of your analysis will be to how you handle such step in your pipeline.

**NOTE**: Be careful about data leakage when standardizing. You should always **fit the scaler on training data only**, then use the same fitted scaler to transform both training and test data. Never fit on test data. This is an important concept and violation of it in your results will result in major deduction in your score.

---

## 5. ML Methods & Evaluation

This section summarizes the ML methods you must apply to your SER features and how to evaluate them. The **weekly roadmap (Section 6)** specifies when to work on each method. For every model below, implement it in MiniLearn (from scratch) **and** apply the scikit-learn equivalent, then compare results.

### 5.1 Required Models

**Supervised (Classification):**

- Logistic Regression, Gaussian Naive Bayes, k-Nearest Neighbors
- Support Vector Machine (try multiple kernels)
- Decision Tree (CART) and Random Forest
- At least one neural network: Dense NN, 1D-CNN, or LSTM (may use TensorFlow/Keras or PyTorch)

**Unsupervised (Clustering):**

- K-Means (k = 8, matching the number of emotions). Evaluate with Adjusted Rand Index and Normalized Mutual Information. Visualize clusters in 2D (PCA or t-SNE) colored by cluster assignment vs. true emotion.
- Optional: Hierarchical Clustering, DBSCAN, or GMMs.

### 5.2 Required Metrics

For every supervised model, report: **Accuracy**, **Precision**, **Recall**, **F1 Score** (per-class and macro/weighted), **Confusion Matrix** (heatmap), and **ROC/AUC** (One-vs-Rest).

### 5.3 Validation Strategy

- Use **Stratified K-Fold Cross-Validation** (k = 5 or 10) for classical models.
- For neural networks, use a held-out validation split.
- Perform hyperparameter tuning (GridSearch or RandomizedSearch) on training folds only.

### 5.4 Comparative Analysis

Build a summary table of all models (accuracy, macro-F1, AUC, best hyperparameters, training time). In your report, discuss which model performed best and why, which emotions are hardest to classify, how MiniLearn compares to scikit-learn, and how different feature sets affect performance.

---

## 6. Weekly Roadmap

| Week      | Topic                                        | Project Tasks                                                                                                                                                                                                                                        | Deliverable                                                                  |
| --------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **4**     | Data Wrangling & Feature Engineering         | Download RAVDESS data, set up environment. Parse filenames and build metadata table. Explore class distributions and audio properties.                                                                                                               | Data loading pipeline + EDA notebook                                         |
| **5**     | Data Wrangling & Feature Engineering (cont.) | Extract audio features (MFCCs, ZCR, RMS, spectral features). Save to CSV. Correlation analysis, feature distributions by emotion, visualizations. Apply feature standardization.                                                                     | Feature module and CSV + EDA notebook                                        |
| **6**     | Regression                                   | Finalize any remaining feature work. Apply a regression model (e.g., linear regression to predict emotional intensity from audio features). Evaluate with regression metrics (MSE, R²). Begin implementing Logistic Regression in MiniLearn.         | MiniLearn Regression (optional: LR draft) + Regression notebook for analysis |
| **7**     | Classification                               | Apply Logistic Regression, Gaussian Naive Bayes, and kNN to SER. Compare MiniLearn vs scikit-learn.                                                                                                                                                  | MiniLearn LR, NB and kNN classifiers + Classification notebook for analysis  |
| **8**     | SVM                                          | Apply SVM with linear, RBF, and polynomial kernels. Tune hyperparameters (C, gamma).                                                                                                                                                                 | MiniLearn SVM + SVM results notebook                                         |
| 9         | Decision Trees & Ensembles                   | Implement **only** CART from scratch in MiniLearn. Apply scikit-learn Decision Tree, Random Forest, and AdaBoost for empirical comparison. Visualize trees and discuss overfitting and generalization.                                               | MiniLearn CART + ensemble analysis notebook                                  |
| **10**    | Model Validation                             | Apply Stratified K-Fold cross-validation to all models. Hyperparameter tuning (GridSearch/RandomizedSearch). Build model comparison table.                                                                                                           | MiniLearn should handle k-fold + Validation notebook                         |
| **11**    | Clustering                                   | Apply K-Means (k = 8). Evaluate with ARI/NMI. Visualize clusters using PCA or t-SNE. Optional: hierarchical clustering, DBSCAN.                                                                                                                      | MiniLearn kMeans + Clustering notebook                                       |
| **12**    | Dimensionality Reduction                     | Apply PCA for feature extraction. Analyze explained variance. Re-run classifiers on reduced features. Optional: non-linear techniques.                                                                                                               | MiniLearn PCA + Dimensionality reduction notebook                            |
| **13–14** | ANN                                          | Implement a simple ANN from scratch in MiniLearn. Apply one off-the-shelf DL model (e.g., Dense NN, 1D-CNN, or LSTM). Compare with classical models. Finalize report and clean repository. _Week 14 may be used for project evaluation/walkthrough._ | MiniLearn ANN + DL notebook + final report                                   |

---

## 7. Setup

### Required Python Packages

You will need: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `librosa`, `soundfile`, `xgboost`.

- I encourage you to use a deep learning framework (e.g., `tensorflow` or `pytorch`) for the ANN portion of the project. This will give you valuable exposure to deep learning concepts and tools that are widely used in both industry and academia. You can also use it for more advanced feature extraction (e.g., pre-trained audio embeddings) if you wish.

### Dataset Download

Go to [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976) and manually download the two audio-only zip files. Extract them into a `data/` folder in your project.

---

## 8. Final Submission

Your final deliverable is a **compiled version of your weekly notebooks** organized into a coherent, report-style document. You do not need to write a separate report — instead, combine and clean up the notebooks you have been building throughout the semester.

Your compiled submission should include:

1. **Introduction** — Brief overview of SER and the RAVDESS dataset.
2. **Your weekly notebooks** — Data exploration, feature engineering, each ML method, clustering, dimensionality reduction, and ANN — organized in logical order with transitions between sections.
3. **Discussion** — Key findings, model comparison highlights, limitations, and what you would do differently.
4. **Conclusion** — Summary of best-performing methods and practical takeaways.
5. **References** — Cite the RAVDESS paper, libraries, and any resources you used.

For every plot and table, include a brief explanation of what it shows and what it means.

> **AI tools are permitted** for writing, editing, and polishing the final report text. However, you must understand and be able to discuss every result and figure during the code walkthrough.

---

## 9. Rubric

### Evaluation Approach

The evaluation will be **qualitative** — you need to **own the code**. The final score will be determined based on the distribution of scores, so the highest standard determines the scale. **105% (5% bonus)** will be given to the best project. Your judgment and care will be evaluated; this is a project, not an assignment, so there is no clear direction on every step. What you bring to the table matters.

### Total: 100 points

| Section                                         | Points | Key Criteria                                                                              |
| ----------------------------------------------- | ------ | ----------------------------------------------------------------------------------------- |
| **A. Data Acquisition, Cleaning & Exploration** | 10     | Correct download, filename parsing, data audit, EDA visualizations, written discussion    |
| **B. Feature Extraction**                       | 10     | Multiple feature types extracted, analysis/visualization, proper standardization          |
| **C. MiniLearn Library**                        | 30     | Package structure, LR, KNN, NB, Decision Tree, metrics module — all from scratch          |
| **D. Supervised Classification**                | 20     | All classical + boosting + at least one NN model applied; MiniLearn vs sklearn comparison |
| **E. Model Evaluation & Validation**            | 10     | All metrics reported, cross-validation, hyperparameter tuning                             |
| **F. Unsupervised / Clustering**                | 10     | K-Means + Hierarchical applied, PCA/t-SNE visualization, ARI/NMI, written analysis        |
| **G. Report Quality & Presentation**            | 10     | Organization, writing quality, critical discussion, code cleanliness                      |

### Code Ownership (applies to all sections)

You must be able to explain every piece of code you submit. I reserve the right to conduct **individual code walkthroughs** at any time. Inability to explain your work will be treated as evidence that it is not your own and may result in a score of **0 for the affected sections**.

---

## 10. Academic Integrity

You may use AI tools for guidance, debugging, learning concepts, and writing/polishing your report. However:

- You must **understand and be able to explain** every line of code and every result.
- Your Git history should show **incremental development**, not a single bulk commit.
- **Code walkthroughs are mandatory** — I will ask you to explain your work.
- Submitting code you cannot explain will be treated as an academic integrity violation.

---

## References

1. Livingstone SR, Russo FA (2018). _The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)._ PLoS ONE 13(5): e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)
2. RAVDESS Dataset on Zenodo: [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976)
3. McFee B, et al. _librosa: Audio and music signal analysis in Python._ [https://librosa.org](https://librosa.org)
4. You may take a look into sample Papers
