# Missing Data Handling Guide

This document describes the missing data in the aggregated dataset for modeling.

## Overview

The aggregated dataset (`aggregated_dataset_for_modeling.xlsx`) contains 46,872 rows with 35 predictors and 1 output variable. **NO PREPROCESSING OR IMPUTATION IS PERFORMED** - missing values are preserved as NaN.

## Missing Data Summary

### 1. Pre-Entry Volume Features (volume_m-24 to volume_m-1)

**Missing Pattern:** Some country-brand combinations may not have data for all 24 months before generic entry.

**Handling Method:** 
- **Missing values are preserved as NaN**
- No imputation performed
- User should decide how to handle these in their model

**Impact:** 
- 24 features created from months -24 to -1
- Missing values indicate periods with no recorded volume data

### 2. Number of Generic Competitors (n_gxs)

**Missing Pattern:** 
- **12,336 missing values (26.32% of records)**
- Missing data occurs when generic competitor information is not available for specific country-brand-month combinations

**Handling Method:**
- **Missing values are preserved as NaN**
- No imputation performed
- User should decide how to handle these (e.g., fill with 0, forward-fill, or use as indicator)

**Impact:**
- This is a significant portion of missing data
- Consider creating a missing indicator feature: `n_gxs_missing = n_gxs.isna()`

### 3. Hospital Rate (hospital_rate)

**Missing Pattern:**
- **1,536 missing values (3.28% of records)**
- Missing data occurs when hospital rate information is not available for specific country-brand combinations

**Handling Method:**
- **Missing values are preserved as NaN**
- No imputation performed
- User should decide how to handle these (e.g., median imputation, mode, or country-specific imputation)

**Impact:**
- Small percentage of missing data
- Could use median, mean, or country/therapeutic area-specific imputation

## Data Quality Summary

### Missing Values by Column:
| Feature | Missing Count | Missing % | Recommended Handling |
|---------|---------------|-----------|----------------------|
| Pre-entry volumes (24 features) | Variable | Variable | Zero fill, forward-fill, or interpolation |
| n_gxs | 12,336 | 26.32% | Zero fill or missing indicator |
| hospital_rate | 1,536 | 3.28% | Median/mode imputation or country-specific |

## Recommendations for Model Training

1. **Pre-entry Volume Features:**
   - Option 1: Fill missing with 0 (assume no volume)
   - Option 2: Forward-fill or backward-fill within country-brand groups
   - Option 3: Interpolation for time series
   - Option 4: Create missing indicators

2. **n_gxs (Generic Competitors):**
   - Option 1: Fill with 0 (assume no generics)
   - Option 2: Forward-fill within country-brand groups
   - Option 3: Create binary indicator: `has_n_gxs_data = ~n_gxs.isna()`
   - Option 4: Use separate model for missing vs non-missing

3. **hospital_rate:**
   - Option 1: Fill with median (4.95) or mean
   - Option 2: Country-specific median
   - Option 3: Therapeutic area-specific median
   - Option 4: Mode imputation

4. **Model Validation:**
   - When evaluating model performance, consider the impact of missing values
   - Test model robustness by comparing performance on records with vs without missing values
   - Consider using models that handle missing values natively (e.g., XGBoost, LightGBM)

## Code Reference

The dataset is created in `create_aggregated_dataset.py` with **NO IMPUTATION**:

```python
# All merges use how='left' - missing values preserved as NaN
dataset = dataset.merge(avg_vol, on=['country', 'brand_name'], how='left')
dataset = dataset.merge(pre_vol_pivot, on=['country', 'brand_name'], how='left')
dataset = dataset.merge(gen_train[['country', 'brand_name', 'months_postgx', 'n_gxs']], 
                        on=['country', 'brand_name', 'months_postgx'], how='left')
dataset = dataset.merge(med_train[...], on=['country', 'brand_name'], how='left')
```

## Summary

- **Total missing values:** ~13,872+ (across all features)
- **Imputation performed:** None
- **Missing values preserved as:** NaN
- **User responsibility:** Handle missing values according to their modeling approach

## Notes

- All other features (avg_vol, months_postgx, month, medicine info features except hospital_rate, country, brand_name) have no missing values
- The output variable (volume) has no missing values as it's the target we're predicting
- Missing data patterns may be informative - consider exploring relationships between missingness and other features
- Some models (XGBoost, LightGBM, CatBoost) can handle missing values natively
