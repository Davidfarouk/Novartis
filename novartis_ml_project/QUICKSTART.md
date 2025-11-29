# Quick Start Guide

## Setup

### Windows (Easiest - Using Batch Files)

1. **First time setup:**
   ```bash
   setup.bat
   ```
   This will create the virtual environment and install all dependencies.

2. **Run pipeline:**
   ```bash
   run.bat
   ```

### Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data structure:**
   Ensure the following directory structure exists:
   ```
   novartis/
   ├── SUBMISSION/
   │   ├── Data files/
   │   │   ├── TRAIN/
   │   │   │   ├── df_volume_train.csv
   │   │   │   ├── df_generics_train.csv
   │   │   │   └── df_medicine_info_train.csv
   │   │   └── TEST/
   │   │       ├── df_volume_test1.csv
   │   │       ├── df_generics_test1.csv
   │   │       └── df_medicine_info_test1.csv
   │   └── Submission example/
   │       └── submission_example.csv
   └── novartis_ml_project/
       └── ...
   ```

## Run Pipeline

**Complete pipeline (training + prediction):**
```bash
cd novartis_ml_project
python main.py
```

This will:
1. Load and merge all data
2. Engineer features
3. Train LightGBM and XGBoost models with cross-validation
4. Generate predictions on test data
5. Create submission file

## Output Files

After running, you'll find:

- **`outputs/submission.csv`** - Final submission file
- **`outputs/cv_results_lgb.csv`** - LightGBM cross-validation results
- **`outputs/cv_results_xgb.csv`** - XGBoost cross-validation results
- **`outputs/feature_importance_*.csv`** - Feature importance rankings
- **`models/lgb_model.txt`** - Trained LightGBM model
- **`models/xgb_model.json`** - Trained XGBoost model
- **`logs/training_log_*.txt`** - Complete training log with loss tracking

## Training Logs

All training progress is logged to `logs/training_log_YYYYMMDD_HHMMSS.txt`:
- Loss at each iteration
- Validation scores per fold
- Feature importance
- Final metrics

## Monitoring Training

The training log file contains:
- Cross-validation scores for each fold
- Bucket 1 and Bucket 2 PE scores
- Feature importance rankings
- Model performance metrics

## Troubleshooting

**Issue: Module not found**
- Ensure you're in the `novartis_ml_project` directory
- Check that virtual environment is activated
- Verify all dependencies are installed

**Issue: Data file not found**
- Check that `SUBMISSION` folder is in the parent directory
- Verify file paths in error messages

**Issue: Out of memory**
- Reduce `n_estimators` in train.py
- Use smaller `num_leaves` for LightGBM

