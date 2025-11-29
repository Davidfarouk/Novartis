# Project Summary

## What Was Built

A complete, production-ready machine learning pipeline for the Novartis Datathon 2025 competition.

## Complete Project Structure

```
novartis_ml_project/
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md             # Quick start guide
├── PROJECT_SUMMARY.md        # This file
│
├── src/                      # Source code modules
│   ├── data_loader.py        # Data loading and merging
│   ├── feature_engineering.py # Feature creation (11 engineered features)
│   ├── metric.py             # Competition metric (exact match to official)
│   ├── train.py              # Model training with full logging
│   └── predict.py            # Prediction pipeline
│
├── data/                      # Processed data (generated)
│   ├── train_merged.csv
│   └── auxiliary_train.csv
│
├── models/                    # Saved models (generated)
│   ├── lgb_model.txt
│   ├── xgb_model.json
│   └── label_encoders.pkl
│
├── logs/                      # Training logs (generated)
│   └── training_log_*.txt    # Complete loss tracking
│
└── outputs/                   # Results (generated)
    ├── submission.csv
    ├── cv_results_*.csv
    └── feature_importance_*.csv
```

## Key Components

### 1. Data Loading (`src/data_loader.py`)
- Loads all training and test data files
- Merges volume, generics, and medicine info
- Computes auxiliary data: `avg_vol`, `mean_erosion`, `bucket`

### 2. Feature Engineering (`src/feature_engineering.py`)
- **11 Engineered Features** from pre-entry data (months -12 to -1):
  - avg_vol, vol_std, vol_min, vol_max, vol_last
  - vol_cv, vol_range_ratio, vol_ratio_last
  - vol_trend, vol_momentum, n_gxs_at_entry
- Label encoding for categorical features
- Missing value handling (n_gxs=0, hospital_rate=median)
- Sample weight computation (bucket × time period)

### 3. Competition Metric (`src/metric.py`)
- **Exact match** to official `metric_calculation.py`
- Scenario 1 (Phase 1-a): Predict months 0-23
- Scenario 2 (Phase 1-b): Predict months 6-23
- Bucket-weighted final score (Bucket 1 = 2x weight)

### 4. Model Training (`src/train.py`)
- **LightGBM** and **XGBoost** ensemble
- **GroupKFold** cross-validation (5 folds, split by country-brand)
- **Sample weights**: Bucket (2x for Bucket 1) × Time period (2.5x for months 0-5)
- **Full logging**: Loss at each iteration, CV scores, feature importance
- Early stopping with validation monitoring

### 5. Prediction Pipeline (`src/predict.py`)
- Prepares test features (same as training)
- Adds Scenario 2 actual data features (months 0-5)
- Ensemble prediction (average of LightGBM + XGBoost)
- Generates submission CSV

## Training Logs

All training is logged to `logs/training_log_YYYYMMDD_HHMMSS.txt`:

```
- Loss at each iteration (both models)
- Cross-validation scores per fold
- Bucket 1 and Bucket 2 PE scores
- Feature importance rankings
- Model performance metrics
- Complete error tracking
```

## How to Use

### Quick Start:
```bash
cd novartis_ml_project
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

### What Happens:
1. **Data Loading**: Loads and merges all data files
2. **Feature Engineering**: Creates 11 engineered features
3. **Model Training**: Trains LightGBM + XGBoost with 5-fold CV
4. **Logging**: Saves complete training log with loss tracking
5. **Prediction**: Generates predictions on test data
6. **Submission**: Creates `outputs/submission.csv`

## Output Files

After running, you get:

1. **`outputs/submission.csv`** - Final submission (7,488 rows)
2. **`outputs/cv_results_lgb.csv`** - LightGBM cross-validation scores
3. **`outputs/cv_results_xgb.csv`** - XGBoost cross-validation scores
4. **`outputs/feature_importance_*.csv`** - Feature rankings
5. **`models/lgb_model.txt`** - Trained LightGBM model
6. **`models/xgb_model.json`** - Trained XGBoost model
7. **`logs/training_log_*.txt`** - **Complete training log with loss tracking**

## Features Implemented

✅ Exact competition metric  
✅ Training logs with loss tracking  
✅ 11 engineered features  
✅ Scenario 1 & 2 handling  
✅ Bucket-weighted training  
✅ Time-weighted training (months 0-5 prioritized)  
✅ Ensemble model (LightGBM + XGBoost)  
✅ GroupKFold CV (split by country-brand)  
✅ Missing value handling  
✅ Categorical encoding with mapping preservation  
✅ Feature importance tracking  
✅ Complete error tracking and logging  

## Model Configuration

- **LightGBM**: 2000 estimators, LR=0.03, early stopping
- **XGBoost**: 2000 estimators, LR=0.03
- **Ensemble**: Average of both predictions
- **CV**: 5-fold GroupKFold (by country-brand)
- **Weights**: Bucket (1.0 or 2.0) × Period (0.5, 1.0, or 2.5)

## Next Steps

1. Run `python main.py` to train and predict
2. Check `logs/training_log_*.txt` for loss tracking
3. Review `outputs/cv_results_*.csv` for validation scores
4. Submit `outputs/submission.csv` to competition

## Notes

- All paths are relative and work from project root
- Training logs capture every iteration
- Models are saved for reuse
- Feature encoders are saved for consistency
- Complete reproducibility with random seeds

