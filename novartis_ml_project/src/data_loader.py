"""
Data Loading and Preprocessing
Loads all data files and creates base datasets
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_training_data(data_dir=None):
    """Load all training data files"""
    if data_dir is None:
        # Default path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'SUBMISSION', 'Data files', 'TRAIN')
    
    print("=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)
    
    # Load data files
    vol_train = pd.read_csv(os.path.join(data_dir, 'df_volume_train.csv'))
    gen_train = pd.read_csv(os.path.join(data_dir, 'df_generics_train.csv'))
    med_train = pd.read_csv(os.path.join(data_dir, 'df_medicine_info_train.csv'))
    
    print(f"Volume train: {vol_train.shape}")
    print(f"Generics train: {gen_train.shape}")
    print(f"Medicine info train: {med_train.shape}")
    
    return vol_train, gen_train, med_train

def load_test_data(data_dir=None):
    """Load all test data files"""
    if data_dir is None:
        # Default path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.dirname(base_dir)
        data_dir = os.path.join(parent_dir, 'SUBMISSION', 'Data files', 'TEST')
    
    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    
    vol_test = pd.read_csv(os.path.join(data_dir, 'df_volume_test1.csv'))
    gen_test = pd.read_csv(os.path.join(data_dir, 'df_generics_test1.csv'))
    med_test = pd.read_csv(os.path.join(data_dir, 'df_medicine_info_test1.csv'))
    
    # Submission template - Submission example is at the same level as Data files
    submission_base_dir = os.path.dirname(os.path.dirname(data_dir))  # Go up to SUBMISSION
    submission_dir = os.path.join(submission_base_dir, 'Submission example')
    submission_file = os.path.join(submission_dir, 'submission_example.csv')
    
    # Check if file exists, if not try submission_template.csv
    if not os.path.exists(submission_file):
        submission_file = os.path.join(submission_dir, 'submission_template.csv')
    
    if not os.path.exists(submission_file):
        print(f"WARNING: Submission template not found at {submission_file}")
        print("Creating empty submission template from test data structure...")
        # Create a basic template from test data
        submission = vol_test[['country', 'brand_name', 'months_postgx']].copy()
        submission['volume'] = 0.0
    else:
        submission = pd.read_csv(submission_file)
    
    print(f"Volume test: {vol_test.shape}")
    print(f"Generics test: {gen_test.shape}")
    print(f"Medicine info test: {med_test.shape}")
    print(f"Submission template: {submission.shape}")
    
    return vol_test, gen_test, med_test, submission

def merge_training_data(vol_train, gen_train, med_train):
    """Merge all training data"""
    print("\n" + "=" * 80)
    print("MERGING TRAINING DATA")
    print("=" * 80)
    
    # Start with volume data
    df = vol_train.copy()
    print(f"Starting with volume data: {df.shape}")
    
    # Add generics data
    df = df.merge(gen_train[['country', 'brand_name', 'months_postgx', 'n_gxs']], 
                  on=['country', 'brand_name', 'months_postgx'], how='left')
    print(f"After adding generics: {df.shape}")
    
    # Add medicine info
    df = df.merge(med_train[['country', 'brand_name', 'ther_area', 'main_package', 
                             'hospital_rate', 'biological', 'small_molecule']], 
                  on=['country', 'brand_name'], how='left')
    print(f"After adding medicine info: {df.shape}")
    
    return df

def compute_auxiliary_data(vol_train):
    """Compute avg_vol, mean_erosion, and bucket"""
    print("\n" + "=" * 80)
    print("COMPUTING AUXILIARY DATA")
    print("=" * 80)
    
    # Compute avg_vol (months -12 to -1)
    pre_entry = vol_train[(vol_train['months_postgx'] >= -12) & (vol_train['months_postgx'] <= -1)]
    avg_vol = pre_entry.groupby(['country', 'brand_name'])['volume'].mean().reset_index()
    avg_vol.columns = ['country', 'brand_name', 'avg_vol']
    print(f"Computed avg_vol for {len(avg_vol)} country-brand combinations")
    
    # Compute mean_erosion (months 0-23)
    post_entry = vol_train[(vol_train['months_postgx'] >= 0) & (vol_train['months_postgx'] <= 23)]
    post_merged = post_entry.merge(avg_vol, on=['country', 'brand_name'], how='left')
    post_merged['normalized_volume'] = post_merged['volume'] / post_merged['avg_vol']
    
    mean_erosion = post_merged.groupby(['country', 'brand_name'])['normalized_volume'].mean().reset_index()
    mean_erosion.columns = ['country', 'brand_name', 'mean_erosion']
    print(f"Computed mean_erosion for {len(mean_erosion)} country-brand combinations")
    
    # Compute bucket
    auxiliary = avg_vol.merge(mean_erosion, on=['country', 'brand_name'], how='left')
    auxiliary['bucket'] = (auxiliary['mean_erosion'] > 0.25).astype(int) + 1
    # Bucket 1 = high erosion (mean_erosion <= 0.25), Bucket 2 = lower erosion (> 0.25)
    
    print(f"\nBucket distribution:")
    print(auxiliary['bucket'].value_counts().sort_index())
    
    return auxiliary

if __name__ == "__main__":
    # Load data
    vol_train, gen_train, med_train = load_training_data()
    
    # Merge
    df_train = merge_training_data(vol_train, gen_train, med_train)
    
    # Compute auxiliary
    auxiliary = compute_auxiliary_data(vol_train)
    
    # Save (skip if files are open/locked)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        df_train.to_csv(os.path.join(data_dir, 'train_merged.csv'), index=False)
        auxiliary.to_csv(os.path.join(data_dir, 'auxiliary_train.csv'), index=False)
        print("\n" + "=" * 80)
        print("DATA LOADING COMPLETE!")
        print("=" * 80)
        print("Saved files:")
        print(f"  - {os.path.join(data_dir, 'train_merged.csv')}")
        print(f"  - {os.path.join(data_dir, 'auxiliary_train.csv')}")
    except PermissionError:
        print("\n" + "=" * 80)
        print("DATA LOADING COMPLETE!")
        print("=" * 80)
        print("WARNING: Could not save intermediate files - they may be open in another application")
        print("Continuing without saving...")
    except Exception as e:
        print("\n" + "=" * 80)
        print("DATA LOADING COMPLETE!")
        print("=" * 80)
        print(f"WARNING: Could not save intermediate files: {e}")
        print("Continuing without saving...")

