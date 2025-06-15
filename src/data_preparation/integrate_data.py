"""
Module for integrating cleaned lab and clinical data.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
# Add parent directory to sys.path to find config.py one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import utils

log = logging.getLogger(__name__)

def integrate_data(
        clinical_data: pd.DataFrame = None,
        lab_data: pd.DataFrame = None,
        missing_threshold: float = None, 
        remove_tests: list[str] = None, 
        save: bool = False, 
        verbose: bool = False
    ) -> pd.DataFrame:
    """
    Integrate lab and clinical data after filtering and preprocessing.

    Args:
        missing_threshold (float): Threshold (%) to filter columns with missing data.
        remove_tests (list): List of lab test abbreviations to exclude.
        save (bool): Whether to save the cleaned file. Default True.
        verbose (bool): Whether to log in DEBUG mode. Default False.

    Returns:
        pd.DataFrame: Merged DataFrame after integration and cleaning.
    """ 
    # ------------------------------------------------------------------ #
    # Initialize integration parameters
    # ------------------------------------------------------------------ #
    if missing_threshold is None:
        missing_threshold = config.MISSING_THRESHOLD

    if remove_tests is None:
        remove_tests = config.REMOVE_TESTS

    # ------------------------------------------------------------------ #
    # Load 
    # ------------------------------------------------------------------ #
    if clinical_data is None:
        clinical_data = utils.load_csv(config.CLEANED_CLINICAL_PATH)
        log.info(f"Loaded clinical data with shape: {clinical_data.shape}")
    if lab_data is None:
        lab_data = utils.load_csv(config.CLEANED_LAB_PATH)
        log.info(f"Loaded lab data with shape: {lab_data.shape}")

    # ------------------------------------------------------------------ #
    # Filter lab tests by missingness threshold
    # ------------------------------------------------------------------ #
    # Count unique cases per lab test
    lab_test_counts = lab_data.groupby(
        ['test_abbr', 'test_name', 'method_number', 'unit'], as_index=False
    )['case_id'].nunique().rename(columns={'case_id': 'num_cases'})

    total_cases = lab_data["case_id"].nunique()
    lab_test_counts["missing_percentage"] = ((total_cases - lab_test_counts["num_cases"]) / total_cases) * 100
    
    # Filter lab tests by missingness threshold
    filtered_lab_tests = lab_test_counts[lab_test_counts["missing_percentage"] < missing_threshold]
    log.info(f"Filtered lab tests to {filtered_lab_tests.shape[0]} by missingness < {missing_threshold}%")

    # ------------------------------------------------------------------ #
    # Filter non useful specified lab tests (list in config REMOVE_TESTS)
    # ------------------------------------------------------------------ #
    filtered_lab_tests = filtered_lab_tests[~filtered_lab_tests["test_abbr"].isin(remove_tests)]
    log.info(f"Filtered lab tests to {filtered_lab_tests.shape[0]} after removing specified tests")

    # Save reference table
    reference_table = filtered_lab_tests[['test_abbr', 'test_name', 'method_number', 'unit']]
    reference_table.to_csv(config.REFERENCE_TABLE_PATH, index=False)
    log.info(f"Saved lab test reference table to {config.REFERENCE_TABLE_PATH}")

    # Filter lab data to keep only filtered tests
    lab_data_filtered = lab_data.merge(
        filtered_lab_tests,
        on=['test_abbr', 'test_name', 'method_number', 'unit'],
        how='inner'
    )
    log.info(f"Lab data filtered shape: {lab_data_filtered.shape}")

    # ------------------------------------------------------------------ #
    # Pivot lab data to wide format (one row per patient_id and case_id)
    # ------------------------------------------------------------------ #
    lab_data_wide = lab_data_filtered.pivot_table(
        index=['patient_id', 'case_id'],
        columns='test_abbr',
        values='numeric_result',
        aggfunc='first'  # first measurement = initial test at admission
    ).reset_index()
    log.info(f"Lab data pivoted to wide format with shape: {lab_data_wide.shape}")

    # ------------------------------------------------------------------ #
    # Merge clinical data and lab data wide
    # ------------------------------------------------------------------ #
    merged_data = pd.merge(
        clinical_data,
        lab_data_wide,
        on=['patient_id', 'case_id'],
        how='inner'
    )
    log.info(f"Merged data shape: {merged_data.shape}")

    # ------------------------------------------------------------------ #
    # Handle duplicates
    # ------------------------------------------------------------------ #
    merged_data = utils.remove_duplicates(merged_data)

    # ------------------------------------------------------------------ #
    # Remove impossible negative values 
    # ------------------------------------------------------------------ #
    # Only lab entries: starting from 9th column
    neg_cols = utils.find_negative_columns(merged_data, start_col=8)
    merged_data = utils.replace_negatives_with_nan(merged_data, neg_cols)

    # ------------------------------------------------------------------ #
    # Re-filter columns by missingness threshold after aggregation
    # ------------------------------------------------------------------ #
    # Filter out columns with > missing_threshold % missing data
    missing_percentage = (merged_data.isnull().sum() / len(merged_data)) * 100
    cols_to_keep = missing_percentage[missing_percentage <= missing_threshold].index
    merged_data = merged_data.loc[:, cols_to_keep]
    log.info(f"Filtered columns to keep those with missingness <= {missing_threshold}%")
    log.info(f"Final merged data shape after filtering columns: {merged_data.shape}")

    # ------------------------------------------------------------------ #
    # Save & finish
    # ------------------------------------------------------------------ #
    if save:
        utils.save_csv(merged_data, config.MERGED_DATA_PATH)
    log.info(f"Saved merged data to {config.MERGED_DATA_PATH}")

    return merged_data


def replace_invalid_values(
        df_in: pd.DataFrame = None, 
        out_path: str = None, 
        save=False, 
        verbose=False) -> pd.DataFrame:
    """
    Clean dataset by replacing biologically implausible or placeholder values.

    Args:
        df_in (pd.DataFrame, optional): Input dataframe. If None, loads from config.MERGED_DATA_PATH.
        save (bool): Whether to save cleaned dataframe to CSV.
        verbose (bool): Enables logging output.
        output_path (str, optional): File path to save cleaned dataframe. Defaults to config.CLEANED_MERGED_DATA_PATH.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # ------------------------------------------------------------------ #
    # Initialize integration parameters 
    # ------------------------------------------------------------------ #
    log.info("Starting to clean merged data...")

    if df_in is None:
        df = utils.load_csv(config.MERGED_DATA_PATH)
    else:
        df = df_in
        log.info("Using provided DataFrame")

    if out_path is None:
        out_path = config.CLEANED_MERGED_DATA_PATH

    cols_to_check = ['INRiH', 'QUHD', 'Hkn', 'Hbn', 'THZn', 'tHb', 
                     'ALAT', 'ASAT', 'CR', 'BIg', 'CRP', 'GGT', 'GL', 'UREA']

    for col in cols_to_check:
        if col not in df.columns:
            if verbose:
                print(f"Warning: column '{col}' not found in dataframe.")

    # ------------------------------------------------------------------ #
    # Remove INRiH = 9999.0 by replacing with NaN 
    # (considered an invalid extreme value)
    # ------------------------------------------------------------------ #
    df['INRiH'] = df['INRiH'].replace(9999.0, np.nan) 
    log.debug("Replaced 9999.0 with NaN in 'INRiH' column")
    
    # ------------------------------------------------------------------ #
    # Remove invalid QUHD values coded as 9999.0 by replacing with NaN
    # (likely a placeholder for missing or error)
    # ------------------------------------------------------------------ #
    df['QUHD'] = df['QUHD'].replace(9999.0, np.nan)
    log.debug("Replaced 9999.0 with NaN in 'QUHD' column")
    
    # ------------------------------------------------------------------ #
    # Correcting misplaced decimal points
    # ------------------------------------------------------------------ #
    df['Hkn'] = df['Hkn'].replace({42.00: 0.42, 36.50: 0.365}) 
    log.debug("Corrected misplaced decimal points in 'Hkn'")

    # ------------------------------------------------------------------ #
    # Replace 0 with NaN in Hbn, Hkn, THZn, and tHb
    # (0 is biologically implausible for these tests)
    # ------------------------------------------------------------------ #
    for col in ['Hbn', 'Hkn', 'THZn', 'tHb']:
        df[col] = df[col].replace(0, np.nan) 

    # ------------------------------------------------------------------ #
    # Replace 0 in ALAT, ASAT, and CR with 2.5
    # (2.5 is half of the threshold value 5, 
    # as indicated in the corresponding text_result column)
    # ------------------------------------------------------------------ #
    for col in ['ALAT', 'ASAT', 'CR']:
        df[col] = df[col].replace(0, 2.5)

    # ------------------------------------------------------------------ #
    # Replace 0 in BIg, CRP, and GGT with 1.5
    # (1.5 is half of the threshold value 3, 
    # as indicated in the corresponding text_result column)
    # ------------------------------------------------------------------ #
    for col in ['BIg', 'CRP', 'GGT']:
        df[col] = df[col].replace(0, 1.5)
    
    # ------------------------------------------------------------------ #
    # Replace 0 in GL with 0.10
    # (0.10 is half of the threshold value <0.20, 
    # as indicated in the corresponding text_result column)
    # ------------------------------------------------------------------ #
    df['GL'] = df['GL'].replace(0, 0.10)

    # ------------------------------------------------------------------ #
    # Replace 0 in UREA with 0.25
    # (0.25 is half of the threshold value <0.5, 
    # as indicated in the corresponding text_result column)
    # ------------------------------------------------------------------ #
    df['UREA'] = df['UREA'].replace(0, 0.25)
    log.debug("Replaced zeros with NaN or fixed values in various columns")
    
    # ------------------------------------------------------------------ #
    # Save & finish
    # ------------------------------------------------------------------ #
    if save:
        utils.save_csv(df, out_path)

    log.info("Finished replacing invalid values")

    return df

def filter_adults(
    df: pd.DataFrame,
    age_column: str = "age",
    min_age: int = 18,
    max_age: int = 120,
    save: bool = True,
    save_path: str = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Filters the DataFrame to include only adult patients within a realistic age range.

    Args:
        df (pd.DataFrame): Input DataFrame with patient data.
        age_column (str): Name of the column containing patient age.
        min_age (int): Minimum age considered adult (default is 18).
        max_age (int): Maximum age considered valid (default is 120).
        save (bool): If True, saves the filtered DataFrame.
        save_path (str): Path where the filtered DataFrame should be saved.
        verbose (bool): If True, enables DEBUG-level logging.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only adults.
    """
    log.info("Filtering adults between ages %d and %d", min_age, max_age)

    initial_count = len(df)
    adults_df = df[(df[age_column] >= min_age) & (df[age_column] <= max_age)].copy()
    filtered_count = len(adults_df)

    log.debug("Initial row count: %d", initial_count)
    log.debug("Row count after filtering: %d", filtered_count)
    log.info("Filtered out %d non-adult records", initial_count - filtered_count)

    if save:
        if not save_path:
            log.warning("Save is True but no save_path provided. Skipping save.")
        else:
            utils.save_csv(adults_df, save_path)

    return adults_df

if __name__ == "__main__":
    utils.configure_logging(verbose=True)

    merged_data = integrate_data(save=False, verbose=True) # Let the function load the data internally the cleaned datasets to merge
    cleaned_merged_data = replace_invalid_values(merged_data, save=False, verbose=True)
    cleaned_merged_data = filter_adults(df=cleaned_merged_data, save=True, save_path=config.CLEANED_MERGED_DATA_PATH, verbose=True)
    utils.generate_summary_statistics(cleaned_merged_data, start_col=4, save_path = config.LAB_TEST_STATISTICS, verbose=False)