# Novartis Datathon 2025 - Training Results & Parameters

**Date:** November 29, 2025  
**Model Version:** Final with Grid Search & Ensemble Optimization  
**Repository:** [https://github.com/Davidfarouk/Novartis](https://github.com/Davidfarouk/Novartis)

## Final Performance Metrics

### Cross-Validation Results (5-Fold GroupKFold)
- **LightGBM Mean PE:** 0.605952
- **XGBoost Mean PE:** 0.689801
- **Ensemble Weight:** 80% LightGBM + 20% XGBoost

### Best Hyperparameters (Grid Search)
Grid search tested 81 combinations (3-fold CV) and found:
```python
{
    'num_leaves': 31,
    'learning_rate': 0.01,
    'reg_alpha': 1.0,
    'reg_lambda': 1.5
}
Best PE Score: 0.606905
```

## Model Parameters

### LightGBM (Optimized)
```python
{
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 3000,
    'learning_rate': 0.01,  # Optimized from grid search
    'num_leaves': 31,       # Optimized from grid search
    'max_depth': 6,
    'min_child_samples': 50,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'reg_alpha': 1.0,      # Optimized from grid search
    'reg_lambda': 1.5,     # Optimized from grid search
    'verbosity': -1,
    'random_state': 42
}
```

### XGBoost
```python
{
    'objective': 'reg:squarederror',
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'max_depth': 6,
    'min_child_weight': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist'
}
```

## Competition Metric Integration

### Sample Weights Applied
- **Time Period Weights:**
  - Months 0-5: 2.5x (competition metric: 0.5 weight)
  - Months 6-11: 1.0x (competition metric: 0.2 weight)
  - Months 12-23: 0.5x (competition metric: 0.1 weight)
- **Bucket Weights:**
  - Bucket 1 (high erosion): 2x
  - Bucket 2 (low erosion): 1x

## Top 10 Features (LightGBM)

1. early_erosion: 132,693.76
2. months_postgx: 60,080.93
3. vol_cv: 28,533.54
4. vol_stability: 27,090.28
5. n_gxs: 24,674.30
6. hospital_rate: 23,216.44
7. ratio_last_3: 18,583.52
8. vol_momentum_12m: 18,133.98
9. vol_trend_12m: 15,955.21
10. vol_trend_24m: 15,769.64

## Submission File

- **Location:** `novartis_ml_project/outputs/submission.csv`
- **Rows:** 7,488 (all populated)
- **Volume Range:** 56.59 to 111,169,147.79
- **Mean Volume:** 1,984,638.30
- **File Size:** ~344 KB

## Training Configuration

- **Cross-Validation:** GroupKFold (n_splits=5) by (country, brand_name)
- **Grid Search:** 81 combinations, 3-fold CV
- **Early Stopping:** 200 rounds
- **Ensemble:** Weighted average (80% LGB, 20% XGB)

## Improvements Made

1. ✅ Implemented grid search for hyperparameter optimization
2. ✅ Adjusted ensemble weights to favor better-performing LightGBM
3. ✅ Integrated competition metric time-period weights
4. ✅ Used MAE loss (matches competition metric structure)

## Files Generated

- `novartis_ml_project/outputs/submission.csv` - Final submission file
- `novartis_ml_project/outputs/cv_results_lgb.csv` - LightGBM CV results
- `novartis_ml_project/outputs/cv_results_xgb.csv` - XGBoost CV results
- `novartis_ml_project/outputs/feature_importance_lgb.csv` - Feature rankings
- `novartis_ml_project/outputs/feature_importance_xgb.csv` - Feature rankings
- `novartis_ml_project/models/lgb_model.txt` - Trained LightGBM model
- `novartis_ml_project/models/xgb_model.json` - Trained XGBoost model

## Project Structure

```
novartis_ml_project/
├── main.py                    # Main pipeline
├── src/
│   ├── data_loader.py        # Data loading functions
│   ├── feature_engineering.py # Feature creation
│   ├── train.py              # Model training with grid search
│   ├── predict.py            # Prediction and submission generation
│   └── metric.py             # Competition metric implementation
├── outputs/
│   ├── submission.csv        # Final predictions
│   └── *.csv                 # CV results and feature importance
└── models/
    ├── lgb_model.txt         # Trained LightGBM
    └── xgb_model.json        # Trained XGBoost
```

## Usage

See `novartis_ml_project/README.md` and `novartis_ml_project/QUICKSTART.md` for detailed instructions.

Quick start:
```bash
cd novartis_ml_project
python main.py
```

