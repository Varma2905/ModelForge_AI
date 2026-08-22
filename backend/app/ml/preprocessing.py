import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

def preprocess_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocesses a pandas DataFrame based on the provided configuration.

    Handles deduplication, missing-value imputation, and outlier removal.
    Feature scaling is intentionally NOT performed here: scaling must be fit
    only on the training split (see training_routes.train_model), otherwise
    statistics from the test/validation rows leak into training. The chosen
    `scaling` method is still recorded on the dataset doc's preprocessing_config
    and applied later, at train time.

    config example:
    {
        "missing": "mean",  # "remove", "mean", "median", "mode", "none"
        "dedupe": True,
        "outlier": "iqr",   # "iqr", "zscore", "none"
        "scaling": "standard"  # "standard", "minmax", "none" — applied at train time, not here
    }
    """
    # Create a copy to prevent modifying original data
    processed_df = df.copy()
    initial_rows = len(processed_df)
    
    summary = {
        "initial_rows": initial_rows,
        "final_rows": initial_rows,
        "missing_imputed": 0,
        "duplicates_removed": 0,
        "outliers_removed": 0,
    }
    
    # 1. Duplicate Removal
    if config.get("dedupe", False):
        before_dedupe = len(processed_df)
        processed_df.drop_duplicates(inplace=True)
        summary["duplicates_removed"] = before_dedupe - len(processed_df)
        processed_df.reset_index(drop=True, inplace=True)
        
    # 2. Missing Value Handling
    missing_method = config.get("missing", "none")
    if missing_method != "none":
        total_missing_before = processed_df.isna().sum().sum()
        
        if missing_method == "remove":
            processed_df.dropna(inplace=True)
            processed_df.reset_index(drop=True, inplace=True)
        else:
            # Impute column by column
            for col in processed_df.columns:
                if processed_df[col].isna().sum() > 0:
                    # Identify if column is numeric
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        if missing_method == "mean":
                            fill_val = processed_df[col].mean()
                        elif missing_method == "median":
                            fill_val = processed_df[col].median()
                        elif missing_method == "mode":
                            # Use first mode value
                            modes = processed_df[col].mode()
                            fill_val = modes.iloc[0] if not modes.empty else 0
                        else:
                            continue
                        processed_df[col] = processed_df[col].fillna(fill_val)
                    else:
                        # Non-numeric columns use mode replacement for any non-remove missing strategy
                        modes = processed_df[col].mode()
                        if not modes.empty:
                            processed_df[col] = processed_df[col].fillna(modes.iloc[0])
                            
        total_missing_after = processed_df.isna().sum().sum()
        summary["missing_imputed"] = int(total_missing_before - total_missing_after)

    # 3. Outlier Detection and Removal
    outlier_method = config.get("outlier", "none")
    if outlier_method != "none" and len(processed_df) > 0:
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            rows_to_keep = pd.Series(True, index=processed_df.index)
            
            if outlier_method == "iqr":
                for col in numeric_cols:
                    q1 = processed_df[col].quantile(0.25)
                    q3 = processed_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Mark outliers (only for non-null values)
                    col_outliers = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
                    rows_to_keep = rows_to_keep & (~col_outliers)
                    
            elif outlier_method == "zscore":
                for col in numeric_cols:
                    col_std = processed_df[col].std()
                    if col_std > 0:
                        col_mean = processed_df[col].mean()
                        z_scores = (processed_df[col] - col_mean) / col_std
                        col_outliers = z_scores.abs() > 3
                        rows_to_keep = rows_to_keep & (~col_outliers)
            
            before_outliers = len(processed_df)
            processed_df = processed_df[rows_to_keep].reset_index(drop=True)
            summary["outliers_removed"] = before_outliers - len(processed_df)

    # 4. Feature Scaling — deferred to train time (see docstring above).
    # We only record which method was requested so /train-model can apply it
    # correctly after the train/test split.
    summary["scaling_deferred_to_training"] = config.get("scaling", "none")

    summary["final_rows"] = len(processed_df)
    return processed_df, summary
