# Missing Data Handling Guide

This document describes how missing data was handled in the aggregated dataset for modeling.

## Overview

The aggregated dataset (`aggregated_dataset_for_modeling.xlsx`) contains 46,872 rows with 35 predictors and 1 output variable. Some features had missing values that required imputation.

## Missing Data Summary

### 1. Pre-Entry Volume Features (volume_m-24 to volume_m-1)

**Missing Pattern:** Some country-brand combinations may not have data for all 24 months before generic entry.

**Handling Method:** 
- Missing values were filled with **0**
- Rationale: If a month has no data, it indicates zero volume for that period

**Impact:** 
- 24 features created from months -24 to -1
- Missing values are rare and represent periods with no recorded volume

### 2. Number of Generic Competitors (n_gxs)

**Missing Pattern:** 
- **12,336 missing values (26.32% of records)**
- Missing data occurs when generic competitor information is not available for specific country-brand-month combinations

**Handling Method:**
- Missing values were filled with **0**
- Rationale: Missing data likely indicates no generic competitors recorded, which is equivalent to 0 competitors

**Impact:**
- This is a significant portion of missing data
- Consider this in model interpretation: when n_gxs = 0, it could mean either:
  - No generic competitors exist, OR
  - Data was not recorded (missing)

### 3. Hospital Rate (hospital_rate)

**Missing Pattern:**
- **1,536 missing values (3.28% of records)**
- Missing data occurs when hospital rate information is not available for specific country-brand combinations

**Handling Method:**
- Missing values were filled with the **median hospital rate: 4.95%**
- Rationale: Median is robust to outliers and represents a typical hospital rate

**Impact:**
- Small percentage of missing data
- Median imputation preserves the distribution better than mean for this feature

## Data Quality Checks

### Before Imputation:
- Total missing values: 13,872+ (across all features)
- Most critical: n_gxs (26.32% missing)

### After Imputation:
- All missing values handled
- Dataset is complete and ready for modeling

## Recommendations for Model Training

1. **Feature Engineering:**
   - Consider creating a binary indicator for missing n_gxs: `n_gxs_missing = (n_gxs == 0) & (original_n_gxs was missing)`
   - This can help the model distinguish between "no generics" vs "missing data"

2. **Model Validation:**
   - When evaluating model performance, consider the impact of imputed values
   - Test model robustness by comparing performance on records with original vs imputed values

3. **Alternative Approaches:**
   - For n_gxs: Consider using forward-fill or backward-fill within country-brand groups
   - For hospital_rate: Could use country-specific or therapeutic area-specific medians instead of global median

## Code Reference

The missing data handling is implemented in `create_aggregated_dataset.py`:

```python
# Fill missing pre-entry volumes with 0
volume_missing = [col for col in volume_cols if col in missing_cols.index]
if volume_missing:
    dataset[volume_missing] = dataset[volume_missing].fillna(0)

# Fill missing n_gxs with 0
if 'n_gxs' in missing_cols.index:
    dataset['n_gxs'] = dataset['n_gxs'].fillna(0)

# Fill missing hospital_rate with median
if 'hospital_rate' in missing_cols.index:
    median_hosp = dataset['hospital_rate'].median()
    dataset['hospital_rate'] = dataset['hospital_rate'].fillna(median_hosp)
```

## Summary Table

| Feature | Missing % | Imputation Method | Value Used |
|---------|-----------|-------------------|------------|
| Pre-entry volumes (24 features) | Variable | Zero fill | 0 |
| n_gxs | 26.32% | Zero fill | 0 |
| hospital_rate | 3.28% | Median imputation | 4.95 |

## Notes

- All other features (avg_vol, months_postgx, month, medicine info features, country, brand_name) had no missing values
- The output variable (volume) has no missing values as it's the target we're predicting
- Missing data patterns may be informative - consider exploring relationships between missingness and other features

