# Novartis Datathon 2025 - Volume Prediction Dataset

This repository contains the aggregated dataset for predicting pharmaceutical volume erosion after generic entry.

## Dataset

**File:** `aggregated_dataset_for_modeling.xlsx`

### Structure
- **46,872 rows** (monthly records for post-generic entry period)
- **35 predictors** + **1 output variable**
- **Months post generic entry:** 0-23

### Excel Sheets
1. **Main Dataset** - Complete dataset with all features and target variable
2. **Metadata** - Column descriptions, sources, and data types
3. **Summary** - Dataset statistics and overview

## Features

### Predictors (35 features)

1. **Pre-entry Volume Features (24):** `volume_m-24` through `volume_m-1`
   - Monthly volumes from 24 months before to 1 month before generic entry
   - Source: `df_volume` (pre-entry period)
   - Type: Numeric (time series)

2. **avg_vol:** Average volume (months -12 to -1)
   - Source: Computed
   - Type: Numeric

3. **months_postgx:** Months post generic entry (0-23)
   - Source: `df_volume`
   - Type: Numeric

4. **month:** Calendar month
   - Source: `df_volume`
   - Type: Categorical

5. **n_gxs:** Number of generic competitors
   - Source: `df_generics`
   - Type: Numeric

6. **Medicine Information (5 features):**
   - `ther_area`: Therapeutic area (Categorical)
   - `main_package`: Main package type (Categorical)
   - `hospital_rate`: Hospital rate % (Numeric)
   - `biological`: Biological drug (Boolean)
   - `small_molecule`: Small molecule drug (Boolean)
   - Source: `df_medicine_info`

7. **country:** Country code (Categorical)
8. **brand_name:** Brand identifier (Categorical)

### Output Variable

- **volume:** Monthly volume for each `months_postgx` (0-23)
  - This is the target variable to predict

## Missing Data

See [MISSING_DATA_HANDLING.md](MISSING_DATA_HANDLING.md) for detailed information on:
- Missing data patterns
- Imputation methods used
- Recommendations for model training

**Quick Summary:**
- `n_gxs`: 26.32% missing → filled with 0
- `hospital_rate`: 3.28% missing → filled with median (4.95)
- Pre-entry volumes: Missing → filled with 0

## Usage

### Creating the Dataset

Run the script to regenerate the aggregated dataset:

```bash
python create_aggregated_dataset.py
```

This will create `aggregated_dataset_for_modeling.xlsx` with all features merged and missing values handled.

### Requirements

```python
pandas
numpy
openpyxl
```

Install with:
```bash
pip install pandas numpy openpyxl
```

## Data Sources

All original data files are in the `SUBMISSION/Data files/` directory:
- Training data: `TRAIN/` folder
- Test data: `TEST/` folder

## Repository Structure

```
.
├── aggregated_dataset_for_modeling.xlsx  # Main dataset
├── create_aggregated_dataset.py          # Script to create dataset
├── MISSING_DATA_HANDLING.md              # Missing data documentation
├── README.md                              # This file
└── SUBMISSION/                            # Original data files
    └── Data files/
        ├── TRAIN/
        └── TEST/
```

## Notes

- The dataset is ready for machine learning model training
- All missing values have been handled
- Features are organized with predictors first, then output variable
- See Metadata sheet in Excel for detailed column information

