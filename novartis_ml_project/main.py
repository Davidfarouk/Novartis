"""
Main Pipeline - Complete ML Workflow
Runs data loading, feature engineering, training, and prediction
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_training_data, load_test_data, merge_training_data, compute_auxiliary_data
from src.feature_engineering import (
    engineer_features, encode_categorical_features, handle_missing_values, 
    select_features, compute_sample_weights
)
from src.train import train_models, setup_logging, log_message

def main():
    """Main execution pipeline"""
    
    # Setup logging
    log_file = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_message("=" * 80, log_file)
    log_message("NOVARTIS DATATHON 2025 - ML PIPELINE", log_file)
    log_message(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log_message("=" * 80, log_file)
    
    try:
        # =====================================================================
        # STEP 1: LOAD DATA
        # =====================================================================
        log_message("\n" + "=" * 80, log_file)
        log_message("STEP 1: LOADING DATA", log_file)
        log_message("=" * 80, log_file)
        
        vol_train, gen_train, med_train = load_training_data()
        vol_test, gen_test, med_test, submission_template = load_test_data()
        
        # Merge training data
        df_train = merge_training_data(vol_train, gen_train, med_train)
        
        # Compute auxiliary data
        auxiliary = compute_auxiliary_data(vol_train)
        
        # Save intermediate data (skip if files are open/locked)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            df_train.to_csv(os.path.join(data_dir, 'train_merged.csv'), index=False)
            auxiliary.to_csv(os.path.join(data_dir, 'auxiliary_train.csv'), index=False)
            log_message(f"Saved intermediate data to {data_dir}", log_file)
        except PermissionError:
            log_message(f"WARNING: Could not save intermediate data - files may be open in another application", log_file)
            log_message(f"Continuing without saving intermediate files...", log_file)
        except Exception as e:
            log_message(f"WARNING: Could not save intermediate data: {e}", log_file)
            log_message(f"Continuing without saving intermediate files...", log_file)
        
        # =====================================================================
        # STEP 2: FEATURE ENGINEERING
        # =====================================================================
        log_message("\n" + "=" * 80, log_file)
        log_message("STEP 2: FEATURE ENGINEERING", log_file)
        log_message("=" * 80, log_file)
        
        # Engineer features
        df_train = engineer_features(df_train, vol_train, gen_train, auxiliary)
        
        # Handle missing values
        df_train, _, _ = handle_missing_values(df_train, fit=True)
        
        # Encode categorical - FIX: Use absolute path to project models dir
        models_dir = os.path.join(base_dir, 'models')
        df_train, _, _ = encode_categorical_features(df_train, fit=True, encoders_dir=models_dir)
        
        # Compute sample weights (before select_features, to ensure bucket is available)
        sample_weights = compute_sample_weights(df_train, auxiliary)
        
        # Select features (after computing weights)
        X_train, y_train, train_info, feature_cols = select_features(df_train)
        
        log_message(f"Final training features: {len(feature_cols)}", log_file)
        log_message(f"Training samples: {len(X_train)}", log_file)
        
        # =====================================================================
        # STEP 3: MODEL TRAINING
        # =====================================================================
        log_message("\n" + "=" * 80, log_file)
        log_message("STEP 3: MODEL TRAINING", log_file)
        log_message("=" * 80, log_file)
        
        # Option to run grid search (set to True for hyperparameter tuning)
        USE_GRID_SEARCH = True  # Enable limited grid search on key parameters
        
        model_lgb, model_xgb, cv_lgb_df, cv_xgb_df = train_models(
            X_train, y_train, train_info, sample_weights, log_file, use_grid_search=USE_GRID_SEARCH
        )
        
        log_message(f"\nTraining completed!", log_file)
        if len(cv_lgb_df) > 0 and 'final_pe' in cv_lgb_df.columns:
            log_message(f"LightGBM CV Mean PE: {cv_lgb_df['final_pe'].mean():.6f}", log_file)
        else:
            log_message("LightGBM CV: No valid scores", log_file)
        if len(cv_xgb_df) > 0 and 'final_pe' in cv_xgb_df.columns:
            log_message(f"XGBoost CV Mean PE: {cv_xgb_df['final_pe'].mean():.6f}", log_file)
        else:
            log_message("XGBoost CV: No valid scores", log_file)
        
        # =====================================================================
        # STEP 4: TEST PREDICTIONS
        # =====================================================================
        log_message("\n" + "=" * 80, log_file)
        log_message("STEP 4: TEST PREDICTIONS", log_file)
        log_message("=" * 80, log_file)
        
        # Import the new prediction functions
        from src.predict import prepare_submission_features, predict_and_generate_submission
        
        # Prepare features for submission rows (not test data rows!)
        X_test, test_info = prepare_submission_features(vol_test, gen_test, med_test, submission_template)
        
        # Make predictions and generate submission
        submission = predict_and_generate_submission(X_test, test_info, submission_template)
        
        log_message(f"\nSubmission generated: {len(submission)} rows", log_file)
        
        # =====================================================================
        # STEP 5: SUMMARY
        # =====================================================================
        log_message("\n" + "=" * 80, log_file)
        log_message("PIPELINE COMPLETE!", log_file)
        log_message("=" * 80, log_file)
        log_message(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        log_message(f"\nOutput files:", log_file)
        log_message(f"  - outputs/submission.csv", log_file)
        log_message(f"  - outputs/cv_results_lgb.csv", log_file)
        log_message(f"  - outputs/cv_results_xgb.csv", log_file)
        log_message(f"  - outputs/feature_importance_lgb.csv", log_file)
        log_message(f"  - outputs/feature_importance_xgb.csv", log_file)
        log_message(f"  - models/lgb_model.txt", log_file)
        log_message(f"  - models/xgb_model.json", log_file)
        log_message(f"  - logs/training_log_{timestamp}.txt", log_file)
        
    except Exception as e:
        log_message(f"\nERROR: {str(e)}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        raise

if __name__ == "__main__":
    main()
