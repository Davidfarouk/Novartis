"""
Create Aggregated Excel File for Modeling
Each row = one brand-month combination (for months 0-23)
All brand information aggregated from all Excel files
NO PREPROCESSING - missing values preserved as NaN
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREATING AGGREGATED DATASET FOR MODELING")
print("=" * 80)
print("Structure: One row per brand-month (months 0-23)")
print("All brand information aggregated from all source files")
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
# 2. COMPUTE AVG_VOL (from months -12 to -1) - Required feature
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
# 3. CREATE PRE-ENTRY VOLUME FEATURES (months -24 to -1)
# =============================================================================
print("\n" + "=" * 80)
print("CREATING PRE-ENTRY VOLUME FEATURES (months -24 to -1)")
print("=" * 80)

# Filter pre-entry data (months -24 to -1)
pre_entry_full = vol_train[(vol_train['months_postgx'] >= -24) & (vol_train['months_postgx'] <= -1)].copy()

# Pivot to create features for each month - one value per country-brand
pre_vol_pivot = pre_entry_full.pivot_table(
    index=['country', 'brand_name'],
    columns='months_postgx',
    values='volume',
    aggfunc='first'  # Use first value if duplicates exist
).reset_index()

# Rename columns to volume_m-24, volume_m-23, ..., volume_m-1
pre_vol_pivot.columns = ['country', 'brand_name'] + [f'volume_m{int(col)}' for col in pre_vol_pivot.columns[2:]]

print(f"Created {len(pre_vol_pivot.columns) - 2} pre-entry volume features")
print(f"Features: {list(pre_vol_pivot.columns[2:])}")

# =============================================================================
# 4. START WITH POST-ENTRY VOLUME DATA (months 0-23) - This is our base
# =============================================================================
print("\n" + "=" * 80)
print("PREPARING MAIN DATASET (Post-entry months 0-23)")
print("=" * 80)

# Filter for post-entry months (0-23) - each row is a brand-month combination
main_data = vol_train[(vol_train['months_postgx'] >= 0) & (vol_train['months_postgx'] <= 23)].copy()

print(f"Main data shape (post-entry): {main_data.shape}")
print(f"Each row represents: one brand-month combination")
print(f"Total brand-month combinations: {len(main_data)}")

# =============================================================================
# 5. AGGREGATE ALL BRAND INFORMATION FROM ALL FILES
# =============================================================================
print("\n" + "=" * 80)
print("AGGREGATING ALL BRAND INFORMATION FROM ALL FILES")
print("=" * 80)

# Start with volume data (this gives us the base structure - one row per brand-month)
dataset = main_data[['country', 'brand_name', 'months_postgx', 'month', 'volume']].copy()
print(f"1. Starting with volume data: {dataset.shape}")
print(f"   Columns: {list(dataset.columns)}")

# Add avg_vol (computed feature)
dataset = dataset.merge(avg_vol, on=['country', 'brand_name'], how='left')
print(f"2. Added avg_vol: {dataset.shape}")

# Add pre-entry volume features (months -24 to -1)
dataset = dataset.merge(pre_vol_pivot, on=['country', 'brand_name'], how='left')
print(f"3. Added pre-entry volumes (24 features): {dataset.shape}")

# Add generics data (n_gxs) - matched by country, brand_name, and months_postgx
dataset = dataset.merge(gen_train[['country', 'brand_name', 'months_postgx', 'n_gxs']], 
                        on=['country', 'brand_name', 'months_postgx'], how='left')
print(f"4. Added generics data (n_gxs): {dataset.shape}")

# Add medicine info (static features - same for all months for each country-brand)
dataset = dataset.merge(med_train[['country', 'brand_name', 'ther_area', 'main_package', 
                                   'hospital_rate', 'biological', 'small_molecule']], 
                        on=['country', 'brand_name'], how='left')
print(f"5. Added medicine info (ther_area, main_package, hospital_rate, biological, small_molecule): {dataset.shape}")

# =============================================================================
# 6. ORGANIZE COLUMNS: Predictors first, then Output
# =============================================================================
print("\n" + "=" * 80)
print("ORGANIZING COLUMNS")
print("=" * 80)

# Get column order: Predictors first, then Output
predictor_cols = []

# 1. Pre-entry volume features (months -24 to -1)
volume_cols = [col for col in dataset.columns if col.startswith('volume_m')]
volume_cols.sort(key=lambda x: int(x.replace('volume_m', '')))  # Sort by month number
predictor_cols.extend(volume_cols)

# 2. avg_vol
predictor_cols.append('avg_vol')

# 3. months_postgx
predictor_cols.append('months_postgx')

# 4. month (calendar)
predictor_cols.append('month')

# 5. n_gxs (generics)
predictor_cols.append('n_gxs')

# 6. Medicine info features
predictor_cols.extend(['ther_area', 'main_package', 'hospital_rate', 'biological', 'small_molecule'])

# 7. country
predictor_cols.append('country')

# 8. brand_name (identifier)
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
print(f"  Predictors ({len(predictor_cols)}): {', '.join(predictor_cols[:5])}... + {len(predictor_cols)-5} more")
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

print(f"\nTotal rows: {len(dataset):,} (one row per brand-month combination)")
print(f"Total columns: {len(dataset.columns)}")
print(f"  - Predictors: {len(predictor_cols)}")
print(f"  - Output: 1 (volume)")

print(f"\nData structure:")
print(f"  - Months post generic entry: {dataset['months_postgx'].min()} to {dataset['months_postgx'].max()}")
print(f"  - Unique countries: {dataset['country'].nunique()}")
print(f"  - Unique brands: {dataset['brand_name'].nunique()}")
print(f"  - Unique country-brand combinations: {dataset.groupby(['country', 'brand_name']).ngroups}")
print(f"  - Average rows per brand: {len(dataset) / dataset['brand_name'].nunique():.1f}")

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
    descriptions = [f'Volume at month {col.replace("volume_m", "")} before generic entry' for col in volume_cols]
    descriptions.extend([
        'Average volume (months -12 to -1)',
        'Months post generic entry',
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
    ])
    
    sources = ['df_volume (pre-entry)'] * len(volume_cols)
    sources.extend(['Computed', 'df_volume', 'df_volume', 'df_generics'])
    sources.extend(['df_medicine_info'] * 5)  # ther_area, main_package, hospital_rate, biological, small_molecule
    sources.extend(['All files', 'All files', 'df_volume (post-entry)'])
    
    data_types = ['Numeric (time series)'] * len(volume_cols)
    data_types.extend(['Numeric', 'Numeric', 'Categorical', 'Numeric'])
    data_types.extend(['Categorical', 'Categorical', 'Numeric', 'Boolean', 'Boolean'])
    data_types.extend(['Categorical', 'Categorical', 'Numeric'])
    
    types = ['Predictor'] * len(predictor_cols) + ['Output']
    
    # Verify lengths match
    assert len(final_cols) == len(descriptions) == len(sources) == len(data_types) == len(types), \
        f"Length mismatch: cols={len(final_cols)}, desc={len(descriptions)}, src={len(sources)}, dtype={len(data_types)}, type={len(types)}"
    
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
            'Months Post Generic Entry Range',
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
            dataset.isnull().sum().sum(),
            f"{dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns)) * 100:.2f}%",
            'No preprocessing - missing values preserved as NaN'
        ]
    })
    
    summary_stats.to_excel(writer, sheet_name='Summary', index=False)
    print(f"✓ Saved Summary: Dataset statistics")

print(f"\n✓ Excel file saved: {output_file}")
print("\nExcel file contains 3 sheets:")
print("  1. Main Dataset - One row per brand-month (months 0-23)")
print("     All brand information aggregated from all source files")
print("     Missing values preserved as NaN (no preprocessing)")
print("  2. Metadata - Column descriptions, sources, and data types")
print("  3. Summary - Dataset statistics and overview")

print("\n" + "=" * 80)
print("AGGREGATION COMPLETE!")
print("=" * 80)
print("\nDataset Structure:")
print(f"  - Each row = one brand-month combination (for months 0-23)")
print(f"  - Each row contains ALL brand information from all source files")
print(f"  - Predictors: {len(predictor_cols)} features")
print(f"  - Output: volume (monthly volume for that brand-month)")
print(f"  - Missing values: Preserved as NaN (no imputation)")
