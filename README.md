# 🎮 Online Games Popularity Prediction

> A machine learning pipeline for predicting the popularity of Steam games using regression and multi-class classification — combining structured data, advanced feature engineering, and NLP.

🌐 [Live App](https://gamerecommendationpredictor.streamlit.app/) &nbsp;|&nbsp;

---

## 👥 Team Members

| Name |
|------|
| Mohamed Sadek Youssef Ali |
| Mahmoud Hossam Eldin Mahmoud Ahmed |
| Youssef Mahmoud Saad |
| Zeyad Salah Hammad Ahmed |
| Abdelhakim Ahmed Abdelhakim Hassan |
| Ahmed Hany Hamdy Hegag |

---

## 📌 Project Overview

This project analyzes online game data sourced from **Steam** to predict game popularity across two milestones:

| Milestone | Task | Target Variable | Type |
|-----------|------|-----------------|------|
| 1 | Regression | `RecommendationCount` | Continuous |
| 2 | Classification | `GamePopularity` | Low / Medium / High |

---

## 🗂️ Project Structure

```
Online-Games-Popularity-Prediction/
│
├── Data/
│   ├── train_data.csv                  # Raw training data from Steam
│   └── processed/                      # Cleaned and transformed datasets
│
├── Models/
│   ├── regression/                     # Regression model scripts
│   └── classification/                 # Classification model scripts
│
├── saved_models/
│   ├── *.pkl                           # Serialized trained models
│   ├── label_encoder.pkl               # Label encoder for GamePopularity
│   └── preprocessing objects           # Scalers, encoders, etc.
│
├── App.py                              # Streamlit app entry point
├── Project.py                          # Main pipeline script
├── notebook_milestone2.py              # Milestone 2 experiments
├── preprocess.py                       # Preprocessing logic
├── predict.py                          # Inference utilities
├── TestScript.py                       # Testing and evaluation
│
├── preprocessor.pkl                    # General preprocessor
├── production_preprocessor.pkl         # Production-ready preprocessor
│
├── requirements.txt
└── README.md
```

---

## 🧠 Milestone 1 — Regression

**Goal:** Predict `RecommendationCount` — a continuous measure of how many users recommended a game on Steam.

> The target variable was **log-transformed** (`log1p`) to handle skewness before training. Metrics are reported in log space.

### Model Configurations

| Model | Key Hyperparameters |
|-------|---------------------|
| **Linear Regression** | Baseline — no regularization |
| **Ridge Regression** | `alpha=1.0` (L2 regularization) |
| **Random Forest** | `n_estimators=300`, `max_features='sqrt'`, `min_samples_split=5`, `min_samples_leaf=2`, `bootstrap=True` |
| **Gradient Boosting** | `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `min_samples_leaf=3` |
| **XGBoost** | `n_estimators=800`, `learning_rate=0.03`, `max_depth=4`, `subsample=0.7`, `colsample_bytree=0.7`, `min_child_weight=5`, `gamma=0.2`, `reg_alpha=0.3`, `reg_lambda=2.0` |

### Results (Test Set — Log Space)

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | 0.535 | — |
| Ridge Regression | 0.687 | 1.77 |
| Random Forest | 0.766 | 1.53 |
| Gradient Boosting | 0.766 | 1.53 |
| XGBoost | 0.768 | 1.53 |

> Train/test split: **80% / 20%**. Evaluation metrics: MSE, RMSE, MAE, R².

---

## 🏷️ Milestone 2 — Classification

**Goal:** Classify games into popularity tiers using the engineered `GamePopularity` label.

### Target Classes (Ordinal Encoding)

| Class | Encoding | Description |
|-------|----------|-------------|
| 🔴 Low | `0` | Limited traction or recommendations |
| 🟡 Medium | `1` | Moderate popularity |
| 🟢 High | `2` | Widely recommended and popular |

> Ordinal encoding was used to preserve natural rank order.

### Model Configurations

| Model | Key Hyperparameters |
|-------|---------------------|
| **Logistic Regression** | `C ∈ {0.01, 0.1, 1.0, 10.0}`, `max_iter=1000`, `class_weight='balanced'` |
| **Random Forest** | `n_estimators ∈ {100, 200, 300}`, `max_features='sqrt'`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight='balanced'` |
| **Gradient Boosting** | `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `min_samples_leaf=3` |
| **XGBoost** | `learning_rate ∈ {0.01, 0.05, 0.1}`, `n_estimators=500`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='mlogloss'` |

### Results

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| Logistic Regression | 79.99% | 79.02% |
| Random Forest | 98.95% | 91.83% |
| Gradient Boosting | 97.25% | 92.27% |
| **XGBoost** | **99.50%** | **92.60%** ✅ |

> **XGBoost achieved the highest test accuracy at 92.6%** with the best balance between train and test performance.

### Training Time Comparison

| Model | Training Time | Inference Time |
|-------|--------------|----------------|
| Logistic Regression | ~10s | ~0.009s |
| Random Forest | ~4s | ~0.046s |
| Gradient Boosting | ~40s | ~0.020s |
| XGBoost | ~6s | ~0.038s |

---

## 🔧 Preprocessing Pipeline

### 1. Data Cleaning
- Dropping duplicate records
- Detecting nulls via `.isna().sum()` and regex `[^\s*$]` for whitespace-only cells
- Smart null filling:
  - `QueryName` ← `ResponseName` (same feature)
  - `PriceCurrency` ← `'USD'` (only value present)
  - `SupportURL`, `SupportEmail`, `Website` ← empty string
  - `DetailedDescription` ← `AboutText` → `ShortDescription` → `'none'`
  - All remaining nulls ← `'none'` placeholder

### 2. Feature Engineering

| Technique | Details |
|-----------|---------|
| Boolean encoding | Binary flags: `ControllerSupport`, `IsFree`, `FreeVerAvail`, etc. → `0/1` |
| Domain extraction | Parsed `SupportURL` and `SupportEmail` using `urlparser`; matched repeated domains (60-40 split) |
| Frequency encoding | Applied to high-cardinality text features |
| Language extraction | Built binary features for most frequent languages + `SupportedLanguagesCount` |
| Date processing | Converted `ReleaseDate` → game age in years; missing dates filled with median |
| Hardware extraction | Regex-extracted min/recommended RAM and CPU for Win/Linux/Mac (supports G/MHz and G/MB units) |
| Binary notice flags | `LegalNotice`, `DRMNotice`, `ExtUserAcctNotice` → has/doesn't have |
| Image flags | `Background`, `HeaderImage` → binary has/doesn't have |

### 3. Feature Interactions

Interactions were created to capture combined effects on `RecommendationCount`:

| Category | Interaction Features |
|----------|---------------------|
| Ownership × Engagement | `owners_players`, `owners_metacritic`, `players_metacritic` |
| Pricing | `price_discount`, `price_owners`, `price_players`, `free_x_owners`, `free_x_players` |
| Content richness | `content_volume` = Screenshots + Movies + DLC + Packages → `content_owners`, `content_players` |
| Achievements | `achievement_owners`, `achievement_players`, `highlighted_achievements_ratio` |
| Platform reach | `platform_count` (Win+Lin+Mac) → `platform_owners`, `platform_players` |
| Genre × Gameplay | `action_multiplayer`, `rpg_achievement`, `strategy_complexity`, `indie_price` |
| Review depth | `review_words` → `reviews_owners`, `reviews_players`, `reviews_metacritic` |

### 4. NLP Processing

Applied to text fields: `DetailedDescrip`, `ShortDescrip` (merged into `AllText`), `AboutText`, and `Reviews`.

**Steps:**
1. Combined `DetailedDescrip` + `ShortDescrip` → `AllText`
2. Cleaned `AllText`: lowercasing, punctuation removal, extra space removal, domain-specific stopword removal (e.g. "game", "player", "world")
3. Extracted `AllText_len` — word count as a richness signal
4. Created binary keyword features for: `multiplayer`, `online`, `co-op`, `zombie`, `action`, `RPG`, `strategy`, `horror`, etc.
5. **TF-IDF** — word-level (unigrams + bigrams) and character-level (3–5 char n-grams)
6. **Sentiment analysis** — NLTK-based scoring (−1 to +1) on `Reviews` and `AboutText`

All NLP features were merged into a unified feature matrix `X_processed` via scikit-learn pipelines.

### 5. Data Transformation & Selection

- **Log transformation** on 59 skewed features (e.g. `PC_RecRam: 95.15`, `SteamSpyOwners: 90.52`)
- **Outlier removal** using IQR
- **Feature selection** via correlation heatmap — dropped ~80 low-correlation features

**Top features by correlation to `RecommendationCount`:**

| Feature | Correlation |
|---------|-------------|
| SteamSpyPlayersVariance | 0.679 |
| SteamSpyPlayersEstimate | 0.666 |
| content_players | 0.662 |
| platform_players | 0.652 |
| category_players | 0.651 |
| owners_players | 0.643 |
| SteamSpyOwnersVariance | 0.632 |
| SteamSpyOwners | 0.621 |

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mohamedsadek12/Game-Recommendation-Predictor
cd Game_Recommendation_Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run App.py
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
streamlit
```

---

## 📁 Saved Artifacts

| File | Description |
|------|-------------|
| `preprocessor.pkl` | Standard preprocessing pipeline |
| `production_preprocessor.pkl` | Optimized pipeline for deployment |
| `saved_models/*.pkl` | All trained and serialized models |
| `label_encoder.pkl` | Encodes/decodes `GamePopularity` classes |

---

## 📄 License

This project is for academic purposes. Data sourced from Steam via public APIs and datasets.
