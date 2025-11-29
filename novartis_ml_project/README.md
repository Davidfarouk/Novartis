# Novartis Datathon 2025 - ML Project

Complete machine learning pipeline for predicting drug volume erosion after generic entry.

## Project Structure

```
novartis_ml_project/
├── data/                    # Processed data files
│   ├── train_merged.csv    # Merged training data
│   └── auxiliary_train.csv # avg_vol, bucket, mean_erosion
├── src/                     # Source code
│   ├── data_loader.py      # Load and merge data
│   ├── feature_engineering.py  # Feature creation
│   ├── metric.py           # Competition metric implementation
│   ├── train.py            # Model training with logging
│   └── predict.py          # Prediction pipeline
├── models/                  # Saved models
│   ├── lgb_model.txt       # LightGBM model
│   ├── xgb_model.json      # XGBoost model
│   └── label_encoders.pkl  # Categorical encoders
├── logs/                    # Training logs
│   └── training_log_*.txt  # Detailed training logs with loss
├── outputs/                 # Predictions and results
│   ├── submission.csv       # Final submission file
│   ├── cv_results_*.csv    # Cross-validation results
│   └── feature_importance_*.csv  # Feature rankings
├── notebooks/               # Analysis notebooks
├── main.py                  # Main pipeline script
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Quick Start

### Option 1: Using Batch Files (Windows - Easiest)

1. **Setup (first time only):**
   ```bash
   setup.bat
   ```

2. **Run Pipeline:**
   ```bash
   run.bat
   ```

### Option 2: Manual Setup

1. **Setup:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Run Complete Pipeline:**
   ```bash
   python main.py
   ```

This will:
- Load and merge all data
- Engineer features
- Train models with cross-validation
- Generate predictions
- Create submission file

## Key Features

- ✅ **Exact Competition Metric** - Matches official metric_calculation.py
- ✅ **Training Logs** - Complete loss tracking in logs/
- ✅ **Feature Engineering** - 11 engineered features from pre-entry data
- ✅ **Scenario Handling** - Both Scenario 1 (0-23) and Scenario 2 (6-23)
- ✅ **Bucket-Weighted Training** - 2x weight for high-erosion drugs
- ✅ **Time-Weighted Training** - Higher weight for months 0-5
- ✅ **Ensemble Model** - LightGBM + XGBoost average
- ✅ **GroupKFold CV** - Split by country-brand (not random rows)
- ✅ **Missing Value Handling** - n_gxs=0, hospital_rate=median

## Training Logs

All training progress is logged to `logs/training_log_YYYYMMDD_HHMMSS.txt`:

- **Loss Tracking**: Loss at each iteration for both models
- **CV Scores**: PE score for each fold
- **Bucket Scores**: Separate scores for Bucket 1 and Bucket 2
- **Feature Importance**: Rankings from both models
- **Model Performance**: Final metrics and statistics

## Model Configuration

### LightGBM
- Objective: Regression (MAE)
- Learning rate: 0.03
- N estimators: 2000 (with early stopping)
- Max depth: 8
- Feature fraction: 0.8
- Bagging fraction: 0.8

### XGBoost
- Objective: reg:squarederror
- Learning rate: 0.03
- N estimators: 2000
- Max depth: 8
- Subsample: 0.8
- Colsample bytree: 0.8

## Features

### Direct Features
- `months_postgx`: Time since generic entry
- `n_gxs`: Number of generic competitors
- `hospital_rate`: % hospital sales
- `ther_area`: Therapeutic area (label encoded)
- `main_package`: Package type (label encoded)
- `small_molecule`: Drug type (boolean)
- `month`: Calendar month (label encoded)

### Engineered Features (from months -12 to -1)
- `avg_vol`: Mean volume
- `vol_std`: Standard deviation
- `vol_min`, `vol_max`: Min/max volumes
- `vol_last`: Volume at month -1
- `vol_cv`: Coefficient of variation
- `vol_range_ratio`: Range / average
- `vol_ratio_last`: Last / average
- `vol_trend`: Linear regression slope
- `vol_momentum`: Recent vs earlier average
- `n_gxs_at_entry`: Competitors at month 0

## Output Files

After running `main.py`:

1. **`outputs/submission.csv`** - Final submission (7,488 rows)
2. **`outputs/cv_results_lgb.csv`** - LightGBM CV scores
3. **`outputs/cv_results_xgb.csv`** - XGBoost CV scores
4. **`outputs/feature_importance_*.csv`** - Feature rankings
5. **`models/lgb_model.txt`** - Trained LightGBM
6. **`models/xgb_model.json`** - Trained XGBoost
7. **`logs/training_log_*.txt`** - Complete training log

## Metric Details

The competition uses a weighted PE (Prediction Error) score:

- **Scenario 1**: Predict months 0-23 (no post-entry data)
- **Scenario 2**: Predict months 6-23 (with 6 months actual data)

Final score weights Bucket 1 (high erosion) 2x more than Bucket 2.

See `src/metric.py` for exact implementation matching official metric.

