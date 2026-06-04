# 🎮 Online Games Popularity Prediction

> A machine learning pipeline for predicting the popularity of Steam games using regression and multi-class classification techniques — combining structured data, feature engineering, and NLP.

---

## 📌 Project Overview

This project analyzes online game data sourced from **Steam** to predict game popularity across two milestones:

| Milestone | Task | Target Variable |
|-----------|------|-----------------|
| 1 | Regression | `RecommendationCount` (continuous) |
| 2 | Classification | `GamePopularity` (Low / Medium / High) |

---

## 🗂️ Project Structure

```
Online-Games-Popularity-Prediction/
│
├── Data/
│   ├── train_data.csv              # Raw training data from Steam
│   └── processed/                  # Cleaned and transformed datasets
│
├── Models/
│   ├── regression/                 # Regression model scripts
│   └── classification/             # Classification model scripts
│
├── saved_models/
│   ├── *.pkl                       # Serialized trained models
│   ├── label_encoder.pkl           # Label encoder for target variable
│   └── preprocessing objects       # Scalers, encoders, etc.
│
├── App.py                          # Streamlit / Flask app entry point
├── Project.py                      # Main pipeline script
├── notebook_milestone2.py          # Milestone 2 experiments
├── preprocess.py                   # Preprocessing logic
├── predict.py                      # Inference utilities
├── TestScript.py                   # Testing and evaluation
│
├── preprocessor.pkl                # General preprocessor
├── production_preprocessor.pkl     # Production-ready preprocessor
│
├── requirements.txt
└── README.md
```

---

## 🧠 Milestone 1 — Regression

**Goal:** Predict `RecommendationCount` — a continuous measure of how many users recommended a game.

### Models Trained

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Polynomial Regression | Captures non-linear relationships |
| Ridge Regression | L2 regularization to reduce overfitting |
| Random Forest Regressor | Ensemble, handles non-linearity well |
| Gradient Boosting Regressor | Sequential boosting for high accuracy |
| XGBoost Regressor | Optimized gradient boosting |

---

## 🏷️ Milestone 2 — Classification

**Goal:** Predict `GamePopularity` class — a derived categorical label.

### Target Classes

| Class | Description |
|-------|-------------|
| 🔴 Low | Game has limited traction or recommendations |
| 🟡 Medium | Moderate popularity |
| 🟢 High | Widely recommended and popular |

### Models Trained

| Model | Hyperparameter Tuning |
|-------|----------------------|
| Logistic Regression | ✅ |
| Random Forest Classifier | ✅ |
| XGBoost Classifier | ✅ |
| K-Nearest Neighbors (KNN) | ✅ |

All models were tuned using grid/random search cross-validation.

---

## 🔧 Preprocessing Pipeline

### 1. Data Cleaning
- Removing duplicate records
- Handling missing values and replacing empty cells
- Fixing inconsistent data types

### 2. Feature Engineering
- **Boolean encoding** — binary flags for game attributes
- **Frequency encoding** — for high-cardinality categorical features
- **Language extraction** — parsing supported languages
- **Date processing** — release year, age of game, etc.
- **Hardware requirements extraction** — minimum/recommended specs
- **Feature interactions** — combining related features

### 3. NLP Processing

Applied to four text fields:

| Field | Description |
|-------|-------------|
| `DetailedDescrip` | Full game description |
| `ShortDescrip` | Short store description |
| `AboutText` | "About this game" section |
| `Reviews` | User review text |

Techniques applied:
- Text cleaning and normalization
- Custom stopword removal
- **TF-IDF** (word-level and character-level n-grams)
- **Sentiment analysis** (positive/negative/neutral scoring)
- **Keyword extraction**
- Text length and readability features

### 4. Data Transformation
- Log transformation for right-skewed numerical features
- Outlier handling using **IQR (Interquartile Range)**
- Feature selection via **correlation analysis**

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mohamedsadek12/Game-Recommendation-Predictor
cd Online-Games-Popularity-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python App.py
```

### 4. Run Predictions
```bash
python predict.py --input Data/train_data.csv
```

---

## 📦 Requirements

Key dependencies (see `requirements.txt` for full list):

```
scikit-learn
xgboost
pandas
numpy
nltk
textblob
matplotlib
seaborn
joblib
```

---

## 📊 Feature Summary

| Category | Examples |
|----------|----------|
| Game metadata | Genre, price, release date, platform support |
| Hardware specs | Minimum/recommended CPU, GPU, RAM |
| Text features | TF-IDF vectors, sentiment scores, description length |
| Encoded features | Supported languages, developer frequency |
| Engineered features | Game age, interaction terms, log-transformed counts |

---

## 📁 Saved Artifacts

| File | Description |
|------|-------------|
| `preprocessor.pkl` | Standard preprocessing pipeline |
| `production_preprocessor.pkl` | Optimized pipeline for deployment |
| `saved_models/*.pkl` | All trained and serialized models |
| `label_encoder.pkl` | Encodes/decodes `GamePopularity` classes |

---

## 👥 Contributors
Ahmed Hany
Mahmoud Hossam
zeyad salah
yousef mahmoud
Abdelhakim Ahmed
---

## 📄 License

This project is for academic purposes. Data sourced from Steam via public APIs and datasets.
