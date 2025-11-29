"""
Create Aggregated Excel File for Modeling
Each volume (pre-entry AND post-entry) = one separate row
Brand information repeated for each volume row
NO PREPROCESSING - missing values preserved as NaN
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREATING AGGREGATED DATASET FOR MODELING")
print("=" * 80)
print("Structure: Each volume = one separate row")
print("Includes pre-entry volumes (months -24 to -1) AND post-entry volumes (0-23)")
print("=" * 80)

# =============================================================================
# 1. LOAD ALL DATA FILES
# =============================================================================
print("\n" + "=" * 80)
print("LOADING DATA FILES")
print("=" * 80)

# Load training data
vol_train = pd.read_csv('SUBMISSION/Data files/TRAIN/df_volume_train.csv')
gen_train = pd.read_csv('SUBMISSION/Data files/TRAIN/df_generics_train.csv')
med_train = pd.read_csv('SUBMISSION/Data files/TRAIN/df_medicine_info_train.csv')

print(f"Volume train: {vol_train.shape}")
print(f"Generics train: {gen_train.shape}")
print(f"Medicine info train: {med_train.shape}")

# =============================================================================
# 2. PREPARE ALL VOLUME DATA (months -24 to 23) - Each volume = one row
# =============================================================================
print("\n" + "=" * 80)
print("PREPARING ALL VOLUME DATA (months -24 to 23)")
print("=" * 80)

# Filter for all months we need: -24 to 23
all_volumes = vol_train[(vol_train['months_postgx'] >= -24) & (vol_train['months_postgx'] <= 23)].copy()

print(f"Total volume records: {len(all_volumes)}")
print(f"Month range: {all_volumes['months_postgx'].min()} to {all_volumes['months_postgx'].max()}")

# Each row now represents one volume record (one brand-month combination)
# We'll add brand information to each row

# =============================================================================
# 3. COMPUTE AVG_VOL (from months -12 to -1) - Required feature
# =============================================================================
print("\n" + "=" * 80)
print("COMPUTING AVG_VOL (Required Feature)")
print("=" * 80)

# Filter for months -12 to -1 (12 months before generic entry)
pre_entry = vol_train[(vol_train['months_postgx'] >= -12) & (vol_train['months_postgx'] <= -1)]

# Calculate avg_vol per country-brand
avg_vol = pre_entry.groupby(['country', 'brand_name'])['volume'].mean().reset_index()
avg_vol.columns = ['country', 'brand_name', 'avg_vol']

print(f"Computed avg_vol for {len(avg_vol)} country-brand combinations")

# =============================================================================
# 4. START WITH ALL VOLUME DATA - Each volume = one row
# =============================================================================
print("\n" + "=" * 80)
print("STARTING WITH ALL VOLUME DATA")
print("=" * 80)

# Start with all volume data - each row is one volume record
dataset = all_volumes[['country', 'brand_name', 'months_postgx', 'month', 'volume']].copy()
print(f"1. Starting with all volume data: {dataset.shape}")
print(f"   Each row = one volume record (one brand-month combination)")
print(f"   Columns: {list(dataset.columns)}")

# =============================================================================
# 5. AGGREGATE ALL BRAND INFORMATION FROM ALL FILES
# =============================================================================
print("\n" + "=" * 80)
print("AGGREGATING ALL BRAND INFORMATION FROM ALL FILES")
print("=" * 80)

# Add avg_vol (computed feature) - same for all rows of the same brand
dataset = dataset.merge(avg_vol, on=['country', 'brand_name'], how='left')
print(f"2. Added avg_vol: {dataset.shape}")

# Add generics data (n_gxs) - matched by country, brand_name, and months_postgx
dataset = dataset.merge(gen_train[['country', 'brand_name', 'months_postgx', 'n_gxs']], 
                        on=['country', 'brand_name', 'months_postgx'], how='left')
print(f"3. Added generics data (n_gxs): {dataset.shape}")

# Add medicine info (static features - same for all rows of the same brand)
dataset = dataset.merge(med_train[['country', 'brand_name', 'ther_area', 'main_package', 
                                   'hospital_rate', 'biological', 'small_molecule']], 
                        on=['country', 'brand_name'], how='left')
print(f"4. Added medicine info (ther_area, main_package, hospital_rate, biological, small_molecule): {dataset.shape}")

# =============================================================================
# 6. ORGANIZE COLUMNS: Predictors first, then Output
# =============================================================================
print("\n" + "=" * 80)
print("ORGANIZING COLUMNS")
print("=" * 80)

# Get column order: Predictors first, then Output
predictor_cols = []

# 1. avg_vol
predictor_cols.append('avg_vol')

# 2. months_postgx (this tells us which month this volume is for)
predictor_cols.append('months_postgx')

# 3. month (calendar)
predictor_cols.append('month')

# 4. n_gxs (generics)
predictor_cols.append('n_gxs')

# 5. Medicine info features
predictor_cols.extend(['ther_area', 'main_package', 'hospital_rate', 'biological', 'small_molecule'])

# 6. country
predictor_cols.append('country')

# 7. brand_name (identifier)
predictor_cols.append('brand_name')

# Output column
output_col = 'volume'

# Final column order
final_cols = predictor_cols + [output_col]

# Reorder dataset
dataset = dataset[final_cols]

print(f"\nFinal dataset shape: {dataset.shape}")
print(f"Number of predictors: {len(predictor_cols)}")
print(f"Number of output columns: 1 (volume)")
print(f"\nColumn order:")
print(f"  Predictors ({len(predictor_cols)}): {', '.join(predictor_cols)}")
print(f"  Output (1): {output_col}")

# =============================================================================
# 7. CHECK MISSING VALUES (NO IMPUTATION - JUST REPORT)
# =============================================================================
print("\n" + "=" * 80)
print("MISSING VALUES SUMMARY (NO IMPUTATION - PRESERVED AS NaN)")
print("=" * 80)

missing_summary = dataset.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]

if len(missing_cols) > 0:
    print("\nMissing values by column (preserved as NaN):")
    for col, count in missing_cols.items():
        pct = count / len(dataset) * 100
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("No missing values found!")

# =============================================================================
# 8. DATASET SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(f"\nTotal rows: {len(dataset):,} (one row per volume record)")
print(f"Total columns: {len(dataset.columns)}")
print(f"  - Predictors: {len(predictor_cols)}")
print(f"  - Output: 1 (volume)")

print(f"\nData structure:")
print(f"  - Months range: {dataset['months_postgx'].min()} to {dataset['months_postgx'].max()}")
print(f"  - Unique countries: {dataset['country'].nunique()}")
print(f"  - Unique brands: {dataset['brand_name'].nunique()}")
print(f"  - Unique country-brand combinations: {dataset.groupby(['country', 'brand_name']).ngroups}")
print(f"  - Average rows per brand: {len(dataset) / dataset['brand_name'].nunique():.1f}")

# Check distribution by month
print(f"\nVolume records by month type:")
pre_entry_count = len(dataset[dataset['months_postgx'] < 0])
post_entry_count = len(dataset[dataset['months_postgx'] >= 0])
print(f"  Pre-entry volumes (months -24 to -1): {pre_entry_count:,} rows")
print(f"  Post-entry volumes (months 0-23): {post_entry_count:,} rows")

# =============================================================================
# 9. SAVE TO EXCEL
# =============================================================================
print("\n" + "=" * 80)
print("SAVING TO EXCEL")
print("=" * 80)

output_file = 'aggregated_dataset_for_modeling.xlsx'

# Create Excel writer
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Main dataset
    dataset.to_excel(writer, sheet_name='Main Dataset', index=False)
    print(f"✓ Saved Main Dataset: {dataset.shape[0]:,} rows × {dataset.shape[1]} columns")
    
    # Create metadata sheet
    descriptions = [
        'Average volume (months -12 to -1)',
        'Months post generic entry (negative = pre-entry, 0-23 = post-entry)',
        'Calendar month',
        'Number of generic competitors',
        'Therapeutic area',
        'Main package type',
        'Hospital rate (%)',
        'Biological drug (True/False)',
        'Small molecule drug (True/False)',
        'Country code',
        'Brand name',
        'Monthly volume (target variable - OUTPUT)'
    ]
    
    sources = [
        'Computed',
        'df_volume',
        'df_volume',
        'df_generics',
        'df_medicine_info',
        'df_medicine_info',
        'df_medicine_info',
        'df_medicine_info',
        'df_medicine_info',
        'All files',
        'All files',
        'df_volume'
    ]
    
    data_types = [
        'Numeric',
        'Numeric',
        'Categorical',
        'Numeric',
        'Categorical',
        'Categorical',
        'Numeric',
        'Boolean',
        'Boolean',
        'Categorical',
        'Categorical',
        'Numeric'
    ]
    
    types = ['Predictor'] * len(predictor_cols) + ['Output']
    
    metadata = pd.DataFrame({
        'Column': final_cols,
        'Type': types,
        'Source': sources,
        'Data Type': data_types,
        'Description': descriptions
    })
    
    metadata.to_excel(writer, sheet_name='Metadata', index=False)
    print(f"✓ Saved Metadata: Column descriptions and types")
    
    # Create summary statistics
    summary_stats = pd.DataFrame({
        'Statistic': [
            'Total Rows',
            'Total Columns',
            'Predictor Columns',
            'Output Columns',
            'Unique Countries',
            'Unique Brands',
            'Unique Country-Brand Combinations',
            'Months Range',
            'Pre-entry Volume Rows',
            'Post-entry Volume Rows',
            'Missing Values (Total)',
            'Missing Values (%)',
            'Note'
        ],
        'Value': [
            len(dataset),
            len(dataset.columns),
            len(predictor_cols),
            1,
            dataset['country'].nunique(),
            dataset['brand_name'].nunique(),
            dataset.groupby(['country', 'brand_name']).ngroups,
            f"{dataset['months_postgx'].min()} to {dataset['months_postgx'].max()}",
            pre_entry_count,
            post_entry_count,
            dataset.isnull().sum().sum(),
            f"{dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns)) * 100:.2f}%",
            'No preprocessing - missing values preserved as NaN. Each volume = separate row.'
        ]
    })
    
    summary_stats.to_excel(writer, sheet_name='Summary', index=False)
    print(f"✓ Saved Summary: Dataset statistics")

print(f"\n✓ Excel file saved: {output_file}")
print("\nExcel file contains 3 sheets:")
print("  1. Main Dataset - Each volume = one separate row")
print("     Includes pre-entry volumes (months -24 to -1) AND post-entry volumes (0-23)")
print("     All brand information aggregated from all source files")
print("     Missing values preserved as NaN (no preprocessing)")
print("  2. Metadata - Column descriptions, sources, and data types")
print("  3. Summary - Dataset statistics and overview")

print("\n" + "=" * 80)
print("AGGREGATION COMPLETE!")
print("=" * 80)
print("\nDataset Structure:")
print(f"  - Each row = one volume record (one brand-month combination)")
print(f"  - Includes ALL volumes: pre-entry (months -24 to -1) AND post-entry (0-23)")
print(f"  - Each row contains ALL brand information from all source files")
print(f"  - Predictors: {len(predictor_cols)} features")
print(f"  - Output: volume (the volume for that specific month)")
print(f"  - Missing values: Preserved as NaN (no imputation)")
