"""
Prediction Pipeline - FIXED VERSION
Makes predictions on test data and generates submission file
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_test_data
from src.feature_engineering import (
    create_pre_entry_features, 
    get_n_gxs_at_entry,
    encode_categorical_features, 
    handle_missing_values
)

def prepare_submission_features(vol_test, gen_test, med_test, submission_template):
    """
    CORRECT APPROACH: Create features for submission rows (months 0-23)
    Based on pre-entry test data (months -24 to -1)
    """
    print("=" * 80)
    print("PREPARING SUBMISSION FEATURES (CORRECT APPROACH)")
    print("=" * 80)
    
    # Step 1: Compute auxiliary data from TEST pre-entry volumes
    print("\nStep 1: Computing auxiliary data for test brands...")
    pre_entry_test = vol_test[(vol_test['months_postgx'] >= -12) & (vol_test['months_postgx'] <= -1)]
    avg_vol_test = pre_entry_test.groupby(['country', 'brand_name'])['volume'].mean().reset_index()
    avg_vol_test.columns = ['country', 'brand_name', 'avg_vol']
    print(f"  Computed avg_vol for {len(avg_vol_test)} test brands")
    
    # Step 2: Compute pre-entry features from TEST historical data
    print("\nStep 2: Computing pre-entry features...")
    country_brand_list = avg_vol_test[['country', 'brand_name']].values.tolist()
    pre_features = create_pre_entry_features(vol_test, country_brand_list)
    print(f"  Created features for {len(pre_features)} brands")
    
    # Step 3: Get n_gxs at entry
    n_gxs_entry = get_n_gxs_at_entry(gen_test, country_brand_list)
    
    # Step 4: Start with SUBMISSION TEMPLATE (these are the rows we need to predict)
    print("\nStep 3: Using submission template as base...")
    df_submission = submission_template.copy()
    print(f"  Submission rows: {len(df_submission)}")
    print(f"  Months to predict: {sorted(df_submission['months_postgx'].unique())}")
    
    # Step 5: Add static features (medicine info)
    print("\nStep 4: Adding static features...")
    df_submission = df_submission.merge(
        med_test[['country', 'brand_name', 'ther_area', 'main_package', 
                  'hospital_rate', 'biological', 'small_molecule']], 
        on=['country', 'brand_name'], how='left'
    )
    
    # Step 6: Add time-varying features (n_gxs for each month)
    df_submission = df_submission.merge(
        gen_test[['country', 'brand_name', 'months_postgx', 'n_gxs']], 
        on=['country', 'brand_name', 'months_postgx'], how='left'
    )
    
    # Step 7: Add pre-entry features (same for all post-entry months of a brand)
    df_submission = df_submission.merge(avg_vol_test, on=['country', 'brand_name'], how='left')
    df_submission = df_submission.merge(pre_features, on=['country', 'brand_name'], how='left')
    df_submission = df_submission.merge(n_gxs_entry, on=['country', 'brand_name'], how='left')
    
    # Step 8: Compute ratio features
    df_submission['ratio_last_3'] = df_submission['vol_last_3_mean'] / df_submission['avg_vol'].replace(0, np.nan)
    df_submission['ratio_last_6'] = df_submission['vol_last_6_mean'] / df_submission['avg_vol'].replace(0, np.nan)
    df_submission['ratio_last_12'] = df_submission['vol_last_12_mean'] / df_submission['avg_vol'].replace(0, np.nan)
    
    # Step 9: Add month name (from submission or infer)
    if 'month' not in df_submission.columns:
        # We don't have actual month names for predictions, use a placeholder
        df_submission['month'] = 'UNKNOWN'
    
    # Step 10: Add early_erosion feature for Scenario 2
    # For Scenario 2 brands (starting at month 6), we could use actual months 0-5 if available
    # For now, leave as NaN (model will handle it)
    df_submission['early_erosion'] = np.nan
    scenario2_brands = df_submission[df_submission['months_postgx'] == 6].groupby(['country', 'brand_name']).size()
    print(f"\n  Scenario 2 brands detected: {len(scenario2_brands)}")
    
    # Step 11: Handle missing values
    df_submission, _, _ = handle_missing_values(df_submission, fit=False)
    
    # Step 12: Encode categorical
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    encoders_dir = os.path.join(base_dir, 'models')
    _, df_submission, _ = encode_categorical_features(
        df_train=None, df_test=df_submission, fit=False, encoders_dir=encoders_dir
    )
    
    # Step 13: Select features in correct order
    feature_cols = [
        'months_postgx', 'n_gxs', 'hospital_rate', 'ther_area', 'main_package', 
        'small_molecule', 'month', 'avg_vol', 'vol_std', 'vol_cv', 
        'vol_trend_12m', 'vol_trend_24m', 'vol_momentum_6m', 'vol_momentum_12m',
        'ratio_last_3', 'ratio_last_6', 'ratio_last_12', 'vol_stability',
        'n_gxs_at_entry', 'early_erosion'
    ]
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df_submission.columns:
            print(f"  WARNING: Missing feature {col}, adding with NaN")
            df_submission[col] = np.nan
    
    X_test = df_submission[feature_cols].copy()
    test_info = df_submission[['country', 'brand_name', 'months_postgx', 'avg_vol']].copy()
    
    print(f"\n✓ Features prepared: {X_test.shape}")
    print(f"✓ avg_vol range: {test_info['avg_vol'].min():.2f} to {test_info['avg_vol'].max():.2f}")
    print(f"✓ No NaN in avg_vol: {test_info['avg_vol'].notna().all()}")
    
    return X_test, test_info

def predict_and_generate_submission(X_test, test_info, submission_template):
    """
    Make predictions and generate submission file
    """
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    # Load models
    lgb_model_path = os.path.join(models_dir, 'lgb_model.txt')
    xgb_model_path = os.path.join(models_dir, 'xgb_model.json')
    
    model_lgb = lgb.Booster(model_file=lgb_model_path)
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(xgb_model_path)
    
    print("✓ Loaded trained models")
    
    # Predict normalized volume
    y_pred_lgb = model_lgb.predict(X_test)
    y_pred_xgb = model_xgb.predict(X_test)
    
    # Ensemble (80% LightGBM, 20% XGBoost) - LightGBM performs better
    y_pred_normalized = 0.8 * y_pred_lgb + 0.2 * y_pred_xgb
    
    print(f"✓ Normalized predictions range: {y_pred_normalized.min():.4f} to {y_pred_normalized.max():.4f}")
    
    # Convert to actual volume
    volume_pred = y_pred_normalized * test_info['avg_vol'].values
    
    # Ensure non-negative
    volume_pred = np.maximum(volume_pred, 0)
    
    print(f"✓ Volume predictions range: {volume_pred.min():.2f} to {volume_pred.max():.2f}")
    print(f"✓ Volume predictions mean: {volume_pred.mean():.2f}")
    print(f"✓ Non-zero predictions: {(volume_pred > 0).sum()} / {len(volume_pred)}")
    
    # Create submission
    submission = submission_template.copy()
    submission['volume'] = volume_pred
    
    # Save
    outputs_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    output_file = os.path.join(outputs_dir, 'submission.csv')
    
    submission.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved submission to: {output_file}")
    print(f"✓ Submission shape: {submission.shape}")
    
    return submission

if __name__ == "__main__":
    print("This script should be called from the main pipeline")
