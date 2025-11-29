"""
Model Training with Logging
Trains LightGBM and XGBoost models with full logging

COMPETITION METRIC INTEGRATION:
- The competition metric (from SUBMISSION/Metric files/metric_calculation.py) is used for EVALUATION
- Training uses MAE (which better matches the metric's absolute difference components)
- Sample weights are adjusted to emphasize time periods matching the competition metric:
  * Months 0-5: 2.5x weight (competition metric gives 0.5 weight to this period)
  * Months 6-11: 1.0x weight (competition metric gives 0.2 weight)
  * Months 12-23: 0.5x weight (competition metric gives 0.1 weight)
- This ensures the model focuses more on early months, which are most important in the competition metric
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold, ParameterGrid
import os
import json
from datetime import datetime
import sys
import itertools

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metric import compute_final_score

def compute_competition_metric_weights(train_info):
    """
    Compute sample weights based on competition metric structure.
    The competition metric emphasizes:
    - Months 0-5: 0.5 weight (most important)
    - Months 6-11: 0.2 weight
    - Months 12-23: 0.1 weight
    - Months 0-23: 0.2 weight (overall MAE component)
    
    Returns weights that emphasize early months more.
    """
    weights = np.ones(len(train_info))
    
    # Get months_postgx from train_info
    if 'months_postgx' in train_info.columns:
        months = train_info['months_postgx'].values
        
        # Competition metric weights by time period
        # Months 0-5 get highest weight (0.5 in metric)
        mask_0_5 = (months >= 0) & (months <= 5)
        weights[mask_0_5] = 2.5  # Highest weight for early months
        
        # Months 6-11 get medium weight (0.2 in metric)
        mask_6_11 = (months >= 6) & (months <= 11)
        weights[mask_6_11] = 1.0
        
        # Months 12-23 get lower weight (0.1 in metric)
        mask_12_23 = (months >= 12) & (months <= 23)
        weights[mask_12_23] = 0.5
        
        # Pre-entry months (for training) get minimal weight
        mask_pre = months < 0
        weights[mask_pre] = 0.1
    
    return weights

def setup_logging(log_dir='../logs'):
    """Setup logging directory and file"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_log_{timestamp}.txt"
    return log_file

def log_message(message, log_file=None, print_msg=True):
    """Log message to file and console"""
    if print_msg:
        print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

def train_models(X_train, y_train, train_info, sample_weights, log_file=None, use_grid_search=False):
    """
    Train LightGBM and XGBoost models with cross-validation
    """
    log_message("=" * 80, log_file)
    log_message("MODEL TRAINING (SSS CLASS HYPERPARAMETERS)", log_file)
    log_message("=" * 80, log_file)
    
    # Get base directory for file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    auxiliary_path = os.path.join(base_dir, 'data', 'auxiliary_train.csv')
    
    # Run grid search if requested
    if use_grid_search:
        best_lgb_params, best_xgb_params = grid_search_hyperparameters(
            X_train, y_train, train_info, sample_weights, auxiliary_path, log_file, n_folds=3
        )
    else:
        best_lgb_params = None
        best_xgb_params = None
    
    # Prepare data
    log_message(f"\nTraining data shape: {X_train.shape}", log_file)
    log_message(f"Target shape: {y_train.shape}", log_file)
    
    # GroupKFold split by (country, brand_name)
    groups = train_info['country'] + '_' + train_info['brand_name']
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    
    # Compute competition-metric-based weights for time periods
    # This emphasizes early months (0-5) which have 0.5 weight in the competition metric
    metric_weights = compute_competition_metric_weights(train_info)
    # Combine with existing sample weights (bucket-based)
    combined_weights = sample_weights * metric_weights
    log_message(f"Applied competition metric time-period weights", log_file)
    log_message(f"  Months 0-5 weight: 2.5x, Months 6-11: 1.0x, Months 12-23: 0.5x", log_file)
    
    # LightGBM parameters - TUNED FOR REGULARIZATION
    # Using MAE which better matches competition metric (uses absolute differences)
    if best_lgb_params is not None:
        lgb_params = best_lgb_params.copy()
        lgb_params['n_estimators'] = 2000
    else:
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',  # MAE matches competition metric better than RMSE
            'boosting_type': 'gbdt',
            'n_estimators': 3000,
            'learning_rate': 0.02,     # Lower LR for better convergence
            'num_leaves': 31,          # Reduced from 63 to prevent overfitting
            'max_depth': 6,            # Reduced from 8
            'min_child_samples': 50,   # Increased from 20
            'feature_fraction': 0.7,   # Subsample features
            'bagging_fraction': 0.7,   # Subsample data
            'bagging_freq': 1,
            'reg_alpha': 1.0,          # Strong L1 regularization
            'reg_lambda': 1.0,         # Strong L2 regularization
            'verbosity': -1,
            'random_state': 42
        }
    
    # XGBoost parameters - TUNED FOR REGULARIZATION
    if best_xgb_params is not None:
        xgb_params = best_xgb_params.copy()
        xgb_params['n_estimators'] = 2000
    else:
        xgb_params = {
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
    
    # Cross-validation
    cv_results_lgb = []
    cv_results_xgb = []
    fold = 0
    
    log_message("\n" + "=" * 80, log_file)
    log_message("CROSS-VALIDATION", log_file)
    log_message("=" * 80, log_file)
    
    for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
        fold += 1
        log_message(f"\n--- Fold {fold}/{n_splits} ---", log_file)
        
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        weights_fold_train = sample_weights[train_idx]
        weights_fold_val = sample_weights[val_idx]
        
        train_info_fold = train_info.iloc[train_idx]
        val_info_fold = train_info.iloc[val_idx]
        
        # ===== LightGBM =====
        log_message("\nTraining LightGBM...", log_file)
        # Use combined weights (bucket weights * time-period weights)
        weights_fold_train_combined = combined_weights[train_idx]
        weights_fold_val_combined = combined_weights[val_idx]
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train, weight=weights_fold_train_combined)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, weight=weights_fold_val_combined, reference=train_data)
        
        model_lgb = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.log_evaluation(period=200),
                lgb.early_stopping(stopping_rounds=200, verbose=True)
            ]
        )
        
        y_pred_lgb = model_lgb.predict(X_fold_val, num_iteration=model_lgb.best_iteration)
        
        # Metric Calculation (Post-entry only)
        val_info_post_entry = val_info_fold[val_info_fold['months_postgx'] >= 0].copy()
        auxiliary = pd.read_csv(auxiliary_path)
        
        if len(val_info_post_entry) > 0:
            val_post_entry_idx = val_info_fold[val_info_fold['months_postgx'] >= 0].index
            val_post_entry_mask = val_info_fold.index.isin(val_post_entry_idx)
            y_pred_lgb_post = y_pred_lgb[val_post_entry_mask]
            
            val_vol_pred_lgb = y_pred_lgb_post * val_info_post_entry['avg_vol'].values
            val_vol_actual = val_info_post_entry['volume'].values
            
            df_actual = pd.DataFrame({
                'country': val_info_post_entry['country'].values,
                'brand_name': val_info_post_entry['brand_name'].values,
                'months_postgx': val_info_post_entry['months_postgx'].values,
                'volume': val_vol_actual
            })
            
            df_pred_lgb = pd.DataFrame({
                'country': val_info_post_entry['country'].values,
                'brand_name': val_info_post_entry['brand_name'].values,
                'months_postgx': val_info_post_entry['months_postgx'].values,
                'volume': val_vol_pred_lgb
            })
            
            try:
                final_pe_lgb, results_lgb = compute_final_score(df_actual, df_pred_lgb, auxiliary)
                log_message(f"LightGBM Fold {fold} - Final PE: {final_pe_lgb:.6f}", log_file)
                cv_results_lgb.append({'fold': fold, 'final_pe': final_pe_lgb})
            except Exception as e:
                log_message(f"Error computing metric: {e}", log_file)
        
        # ===== XGBoost =====
        log_message("\nTraining XGBoost...", log_file)
        model_xgb = xgb.XGBRegressor(**xgb_params)
        
        # Use combined weights (bucket weights * time-period weights)
        model_xgb.fit(
            X_fold_train, y_fold_train,
            sample_weight=weights_fold_train_combined,
            eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
            verbose=200
        )
        
        y_pred_xgb = model_xgb.predict(X_fold_val)
        
        if len(val_info_post_entry) > 0:
            y_pred_xgb_post = y_pred_xgb[val_post_entry_mask]
            val_vol_pred_xgb = y_pred_xgb_post * val_info_post_entry['avg_vol'].values
            
            df_pred_xgb = pd.DataFrame({
                'country': val_info_post_entry['country'].values,
                'brand_name': val_info_post_entry['brand_name'].values,
                'months_postgx': val_info_post_entry['months_postgx'].values,
                'volume': val_vol_pred_xgb
            })
            
            try:
                final_pe_xgb, results_xgb = compute_final_score(df_actual, df_pred_xgb, auxiliary)
                log_message(f"XGBoost Fold {fold} - Final PE: {final_pe_xgb:.6f}", log_file)
                cv_results_xgb.append({'fold': fold, 'final_pe': final_pe_xgb})
            except Exception as e:
                log_message(f"Error computing metric: {e}", log_file)

    # Summary
    cv_lgb_df = pd.DataFrame(cv_results_lgb)
    cv_xgb_df = pd.DataFrame(cv_results_xgb)
    
    log_message("\n" + "=" * 80, log_file)
    log_message("CROSS-VALIDATION SUMMARY", log_file)
    log_message("=" * 80, log_file)
    
    if len(cv_lgb_df) > 0:
        log_message(f"LightGBM Mean PE: {cv_lgb_df['final_pe'].mean():.6f}", log_file)
    if len(cv_xgb_df) > 0:
        log_message(f"XGBoost Mean PE: {cv_xgb_df['final_pe'].mean():.6f}", log_file)
        
    # Final Training
    log_message("\n" + "=" * 80, log_file)
    log_message("TRAINING FINAL MODELS ON ALL DATA", log_file)
    log_message("=" * 80, log_file)
    
    # Use combined weights for final training
    train_data_full = lgb.Dataset(X_train, label=y_train, weight=combined_weights)
    model_lgb_final = lgb.train(lgb_params, train_data_full, num_boost_round=3000, callbacks=[lgb.log_evaluation(period=200)])
    
    model_xgb_final = xgb.XGBRegressor(**xgb_params)
    model_xgb_final.fit(X_train, y_train, sample_weight=combined_weights, verbose=200)
    
    # Save models
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_lgb_final.save_model(os.path.join(models_dir, 'lgb_model.txt'))
    model_xgb_final.save_model(os.path.join(models_dir, 'xgb_model.json'))
    
    # Feature Importance
    fi_lgb = pd.DataFrame({'feature': X_train.columns, 'importance': model_lgb_final.feature_importance(importance_type='gain')}).sort_values('importance', ascending=False)
    fi_xgb = pd.DataFrame({'feature': X_train.columns, 'importance': model_xgb_final.feature_importances_}).sort_values('importance', ascending=False)
    
    outputs_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    fi_lgb.to_csv(os.path.join(outputs_dir, 'feature_importance_lgb.csv'), index=False)
    fi_xgb.to_csv(os.path.join(outputs_dir, 'feature_importance_xgb.csv'), index=False)
    
    log_message("\nTop 10 LightGBM Features:", log_file)
    log_message(fi_lgb.head(10).to_string(), log_file)
    
    return model_lgb_final, model_xgb_final, cv_lgb_df, cv_xgb_df

def grid_search_hyperparameters(X_train, y_train, train_info, sample_weights, auxiliary_path, log_file=None, n_folds=3):
    """
    Limited grid search on key LightGBM hyperparameters
    Focuses on most impactful parameters to save time
    """
    from sklearn.model_selection import GroupKFold
    
    log_message("\n" + "=" * 80, log_file)
    log_message("GRID SEARCH - LightGBM Hyperparameter Tuning", log_file)
    log_message("=" * 80, log_file)
    
    # Limited grid - focus on most impactful parameters
    param_grid = {
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.02, 0.03],
        'reg_alpha': [0.5, 1.0, 1.5],
        'reg_lambda': [0.5, 1.0, 1.5]
    }
    
    # Generate all combinations
    from itertools import product
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    log_message(f"Testing {len(param_combinations)} parameter combinations with {n_folds}-fold CV", log_file)
    
    # GroupKFold split
    groups = train_info['country'] + '_' + train_info['brand_name']
    gkf = GroupKFold(n_splits=n_folds)
    
    # Compute combined weights
    metric_weights = compute_competition_metric_weights(train_info)
    combined_weights = sample_weights * metric_weights
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for idx, params in enumerate(param_combinations):
        log_message(f"\n[{idx+1}/{len(param_combinations)}] Testing: {params}", log_file)
        
        # Base parameters
        base_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'min_child_samples': 50,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'verbosity': -1,
            'random_state': 42
        }
        
        # Merge with grid search params
        lgb_params = {**base_params, **params}
        lgb_params['n_estimators'] = 2000  # Fixed for grid search
        
        # Cross-validation
        cv_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            weights_fold_train = combined_weights[train_idx]
            weights_fold_val = combined_weights[val_idx]
            
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train, weight=weights_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, weight=weights_fold_val, reference=train_data)
            
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                valid_names=['val'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200, verbose=False)
                ],
                num_boost_round=2000
            )
            
            y_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
            
            # Compute competition metric
            val_info_fold = train_info.iloc[val_idx]
            val_info_post_entry = val_info_fold[val_info_fold['months_postgx'] >= 0].copy()
            
            if len(val_info_post_entry) > 0:
                val_post_entry_mask = val_info_fold.index.isin(val_info_post_entry.index)
                y_pred_post = y_pred[val_post_entry_mask]
                
                val_vol_pred = y_pred_post * val_info_post_entry['avg_vol'].values
                val_vol_actual = val_info_post_entry['volume'].values
                
                df_actual = pd.DataFrame({
                    'country': val_info_post_entry['country'].values,
                    'brand_name': val_info_post_entry['brand_name'].values,
                    'months_postgx': val_info_post_entry['months_postgx'].values,
                    'volume': val_vol_actual
                })
                
                df_pred = pd.DataFrame({
                    'country': val_info_post_entry['country'].values,
                    'brand_name': val_info_post_entry['brand_name'].values,
                    'months_postgx': val_info_post_entry['months_postgx'].values,
                    'volume': val_vol_pred
                })
                
                auxiliary = pd.read_csv(auxiliary_path)
                try:
                    final_pe, _ = compute_final_score(df_actual, df_pred, auxiliary)
                    if not np.isnan(final_pe):
                        cv_scores.append(final_pe)
                except:
                    pass
        
        if len(cv_scores) > 0:
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            log_message(f"  Mean PE: {mean_score:.6f} (+/- {std_score:.6f})", log_file)
            
            results.append({
                'params': params,
                'mean_pe': mean_score,
                'std_pe': std_score
            })
            
            if mean_score < best_score:
                best_score = mean_score
                best_params = params.copy()
                log_message(f"  ✓ New best! PE: {best_score:.6f}", log_file)
    
    log_message("\n" + "=" * 80, log_file)
    log_message(f"BEST PARAMETERS: {best_params}", log_file)
    log_message(f"BEST SCORE: {best_score:.6f}", log_file)
    log_message("=" * 80, log_file)
    
    # Return best params for LightGBM, keep XGBoost default
    best_lgb_params = best_params
    best_xgb_params = None  # Keep XGBoost default params
    
    return best_lgb_params, best_xgb_params

if __name__ == "__main__":
    print("Use: python main.py")
