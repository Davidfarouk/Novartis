"""
Competition Metric Implementation
Exact metric calculation matching official metric_calculation.py
"""

import pandas as pd
import numpy as np

def _compute_pe_phase1a(group: pd.DataFrame) -> float:
    """Compute PE for one (country, brand, bucket) group - Scenario 1"""
    avg_vol = group["avg_vol"].iloc[0]
    if avg_vol == 0 or np.isnan(avg_vol):
        return np.nan

    def sum_abs_diff(month_start: int, month_end: int) -> float:
        """Sum of absolute differences sum(|actual - pred|)."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        return (subset["volume_actual"] - subset["volume_predict"]).abs().sum()
    
    def abs_sum_diff(month_start: int, month_end: int) -> float:
        """Absolute difference of |sum(actuals) - sum(pred)|."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        sum_actual = subset["volume_actual"].sum()
        sum_pred = subset["volume_predict"].sum()
        return abs(sum_actual - sum_pred)

    term1 = 0.2 * sum_abs_diff(0, 23) / (24 * avg_vol)
    term2 = 0.5 * abs_sum_diff(0, 5) / (6 * avg_vol)
    term3 = 0.2 * abs_sum_diff(6, 11) / (6 * avg_vol)
    term4 = 0.1 * abs_sum_diff(12, 23) / (12 * avg_vol)

    return term1 + term2 + term3 + term4

def _compute_pe_phase1b(group: pd.DataFrame) -> float:
    """Compute PE for one (country, brand, bucket) group - Scenario 2"""
    avg_vol = group["avg_vol"].iloc[0]
    if avg_vol == 0 or np.isnan(avg_vol):
        return np.nan

    def sum_abs_diff(month_start: int, month_end: int) -> float:
        """Sum of absolute differences sum(|actual - pred|)."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        return (subset["volume_actual"] - subset["volume_predict"]).abs().sum()
    
    def abs_sum_diff(month_start: int, month_end: int) -> float:
        """Absolute difference of |sum(actuals) - sum(pred)|."""
        subset = group[(group["months_postgx"] >= month_start) & (group["months_postgx"] <= month_end)]
        sum_actual = subset["volume_actual"].sum()
        sum_pred = subset["volume_predict"].sum()
        return abs(sum_actual - sum_pred)

    term1 = 0.2 * sum_abs_diff(6, 23) / (18 * avg_vol)
    term2 = 0.5 * abs_sum_diff(6, 11) / (6 * avg_vol)
    term3 = 0.3 * abs_sum_diff(12, 23) / (12 * avg_vol)
    
    return term1 + term2 + term3

def compute_metric1(df_actual: pd.DataFrame, df_pred: pd.DataFrame, df_aux: pd.DataFrame) -> float:
    """Compute Metric 1 (Phase 1-a) - Scenario 1"""
    merged = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_predict")
    ).merge(df_aux, on=["country", "brand_name"], how="left")

    merged["start_month"] = merged.groupby(["country", "brand_name"])["months_postgx"].transform("min")
    merged = merged[merged["start_month"] == 0].copy()

    # Compute PE for each group - use a list comprehension to avoid index issues
    pe_list = []
    for (country, brand, bucket), group in merged.groupby(["country", "brand_name", "bucket"]):
        pe_value = _compute_pe_phase1a(group)
        pe_list.append({
            'country': country,
            'brand_name': brand,
            'bucket': bucket,
            'PE': pe_value
        })
    
    pe_results = pd.DataFrame(pe_list)
    
    # Safety checks: handle empty DataFrame or missing bucket column
    if len(pe_results) == 0:
        return np.nan
    
    if 'bucket' not in pe_results.columns:
        return np.nan
    
    # Filter out NaN bucket values
    pe_results = pe_results[pe_results['bucket'].notna()].copy()
    
    if len(pe_results) == 0:
        return np.nan

    bucket1 = pe_results[pe_results["bucket"] == 1]
    bucket2 = pe_results[pe_results["bucket"] == 2]

    n1 = bucket1[["country", "brand_name"]].drop_duplicates().shape[0]
    n2 = bucket2[["country", "brand_name"]].drop_duplicates().shape[0]

    if n1 == 0 or n2 == 0:
        return np.nan

    return (2/n1) * bucket1["PE"].sum() + (1/n2) * bucket2["PE"].sum()

def compute_metric2(df_actual: pd.DataFrame, df_pred: pd.DataFrame, df_aux: pd.DataFrame) -> float:
    """Compute Metric 2 (Phase 1-b) - Scenario 2"""
    merged_data = df_actual.merge(
        df_pred,
        on=["country", "brand_name", "months_postgx"],
        how="inner",
        suffixes=("_actual", "_predict")
    ).merge(df_aux, on=["country", "brand_name"], how="left")

    merged_data["start_month"] = merged_data.groupby(["country", "brand_name"])["months_postgx"].transform("min")
    merged_data = merged_data[merged_data["start_month"] == 6].copy()

    # Compute PE for each group - use a list comprehension to avoid index issues
    pe_list = []
    for (country, brand, bucket), group in merged_data.groupby(["country", "brand_name", "bucket"]):
        pe_value = _compute_pe_phase1b(group)
        pe_list.append({
            'country': country,
            'brand_name': brand,
            'bucket': bucket,
            'PE': pe_value
        })
    
    pe_results = pd.DataFrame(pe_list)
    
    # Safety checks: handle empty DataFrame or missing bucket column
    if len(pe_results) == 0:
        return np.nan
    
    if 'bucket' not in pe_results.columns:
        return np.nan
    
    # Filter out NaN bucket values
    pe_results = pe_results[pe_results['bucket'].notna()].copy()
    
    if len(pe_results) == 0:
        return np.nan

    bucket1 = pe_results[pe_results["bucket"] == 1]
    bucket2 = pe_results[pe_results["bucket"] == 2]

    n1 = bucket1[["country", "brand_name"]].drop_duplicates().shape[0]
    n2 = bucket2[["country", "brand_name"]].drop_duplicates().shape[0]

    if n1 == 0 or n2 == 0:
        return np.nan

    return (2/n1) * bucket1["PE"].sum() + (1/n2) * bucket2["PE"].sum()

def compute_final_score(df_actual, df_pred, df_aux):
    """
    Compute final competition score (both scenarios combined)
    """
    # Compute metrics for both scenarios
    metric1_score = compute_metric1(df_actual, df_pred, df_aux)
    metric2_score = compute_metric2(df_actual, df_pred, df_aux)
    
    # Get scenario counts
    actual_grouped = df_actual.groupby(['country', 'brand_name'])['months_postgx'].agg(['min']).reset_index()
    scenario1_count = (actual_grouped['min'] == 0).sum()
    scenario2_count = (actual_grouped['min'] == 6).sum()
    
    # Weighted average (or just average if both scenarios present)
    if not np.isnan(metric1_score) and not np.isnan(metric2_score):
        final_pe = (metric1_score + metric2_score) / 2
    elif not np.isnan(metric1_score):
        final_pe = metric1_score
    elif not np.isnan(metric2_score):
        final_pe = metric2_score
    else:
        final_pe = np.nan
    
    results = {
        'final_pe': final_pe,
        'metric1_pe': metric1_score,
        'metric2_pe': metric2_score,
        'scenario1_count': scenario1_count,
        'scenario2_count': scenario2_count
    }
    
    return final_pe, results

