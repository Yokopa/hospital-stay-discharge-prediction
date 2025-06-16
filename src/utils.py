import pandas as pd
import numpy as np
from collections import Counter
import logging

log = logging.getLogger(__name__)

# -----------------------------------#
# Data loading and saving utilities
# -----------------------------------#

def load_csv(path):
    """
    Load a CSV file into a DataFrame.

    Args:
        path (str): File path to the CSV.

    Returns:
        pd.DataFrame: Loaded data.
    """
    log.info("Loading data from %s", path)
    return pd.read_csv(path)

def save_csv(df, path):
    """
    Save a DataFrame to a CSV file (no index column).

    Args:
        df (pd.DataFrame): Data to save.
        path (str): File path to save the CSV.
    """
    df.to_csv(path, index=False)
    log.info("Saved data to %s", path)

def configure_logging(verbose: bool):
    log.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Prevent duplicate handlers if already configured
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        log.addHandler(handler)

# -----------------------------------#
# Preprocessing utilities
# -----------------------------------#

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.

    args:
        df (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: DataFrame without duplicates.
    """
    dups = df.duplicated().sum()
    if dups:
        df = df.drop_duplicates()
        log.debug("Removed %d duplicate rows.", dups)
    return df

def find_negative_columns(
    df: pd.DataFrame,
    start_col: int = None,
    end_col: int = None
) -> list:
    """
    Identify numeric columns with any negative values between start_col and end_col.

    Args:
        df (pd.DataFrame): Input DataFrame.
        start_col (int or None): Index from which to start checking columns (inclusive). 
                                 If None, starts from the first column.
        end_col (int or None): Index at which to stop checking columns (exclusive).
                               If None, checks until the last column.

    Returns:
        list: List of column names with negative values.
    """
    if start_col is None:
        start_col = 0
    if end_col is None:
        numeric_cols = df.columns[start_col:]
    else:
        numeric_cols = df.columns[start_col:end_col]

    negative_columns = df.loc[:, numeric_cols].columns[
        (df.loc[:, numeric_cols] < 0).any()
    ]
    num_negative_entries = (df.loc[:, negative_columns] < 0).sum().sum()

    log.info(f"Total negative entries in numeric lab tests: {num_negative_entries}")
    log.info(f"Columns with negative values: {list(negative_columns)}")

    return list(negative_columns)

def replace_negatives_with_nan(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Replace negative values in specified columns with NA.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.

    Returns:
        pd.DataFrame: DataFrame with negative values replaced by NaN.
    """
    df.loc[:, columns] = df.loc[:, columns].apply(lambda col: col.map(lambda x: np.nan if x < 0 else x))
    log.info(f"Replaced negative values with Nan in columns: {columns}")
    return df

def generate_summary_statistics(
    df: pd.DataFrame,
    start_col: int = None,
    end_col: int = None,
    save_path: str = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate summary statistics and missing percentage for selected columns.

    Args:
        df (pd.DataFrame): The merged dataset.
        start_col (int, optional): Index to start selecting columns (inclusive). If None, starts at 0.
        end_col (int, optional): Index to stop selecting columns (exclusive). If None, goes to end.
        save_path (str, optional): If provided, saves the summary table as CSV.
        verbose (bool): If True, logs summary shape and top missing columns.

    Returns:
        pd.DataFrame: Summary statistics DataFrame.
    """
    # Select column range
    selected_cols = df.columns[start_col:end_col]
    subset_df = df.loc[:, selected_cols]

    # Calculate missing percentage
    missing_percentage = subset_df.isnull().mean() * 100

    # Summary statistics
    summary_stats = subset_df.describe().T
    summary_stats['missing_percentage'] = missing_percentage

    if verbose:
        import logging
        log = logging.getLogger(__name__)
        log.info(f"Summary statistics shape: {summary_stats.shape}")
        log.info("Top columns by missingness:\n" + str(summary_stats.sort_values('missing_percentage', ascending=False).head(5)))

    # Save if path is provided
    if save_path:
        summary_stats.to_csv(save_path)

    return summary_stats

# -----------------------------------#
# Modeling utilities
# -----------------------------------#
# Imbalance
def compute_scale_pos_weight(y):
    """Compute scale_pos_weight = (neg / pos) for XGBoost."""
    counter = Counter(y)
    neg = counter.get(0, 0)
    pos = counter.get(1, 0)
    return neg / pos if pos > 0 else 1.0  # avoid division by zero