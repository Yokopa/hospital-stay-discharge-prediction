"""
Module for cleaning and preprocessing laboratory and clinical data.
"""
import pandas as pd
import sys
import os
# Add parent directory to sys.path to find config.py one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import utils

import logging
log = logging.getLogger(__name__)

def clean_lab_data(df: pd.DataFrame, save=True, verbose=False) -> pd.DataFrame:
    """
    Cleans laboratory data for further analysis.

    Steps:
    - Renaming columns according to a predefined mapping.
    - Optimizing data types for memory efficiency.
    - Checking and logging missing values before and after cleaning.
    - Fixing known data issues (e.g., filling missing abbreviations).
    - Dropping rows with missing results.
    - Attempting to fill missing numeric results from text results.
    - Removing remaining rows with missing numeric results.
    - Removing duplicate entries.
    - Optionally saving the cleaned data to a CSV file.

    Args:
        df (pd.DataFrame): Raw laboratory data to be cleaned.
        save (bool, optional): If True, saves the cleaned DataFrame to disk. Defaults to True.
        verbose (bool, optional): If True, sets logging to DEBUG level for detailed output. Defaults to False.

    Returns:
        pandas.DataFrame: The cleaned laboratory data.
    """

    log.info("\n=== Starting lab data cleaning ===\n")
    log.debug("Shape after load: %s", df.shape)
    log.debug(f"Number of unique entries per column:\n{df.nunique()}")

    # ------------------------------------------------------------------ #
    # Rename columns 
    # ------------------------------------------------------------------ #    
    df.rename(columns=config.COLUMN_TRANSLATION_LAB, inplace=True)
    log.debug("Renamed columns to: %s", list(df.columns))

    # ------------------------------------------------------------------ #
    # Optimise dtypes
    # ------------------------------------------------------------------ #
    df["numeric_result"] = df["numeric_result"].astype("float32")
    for col in ["patient_id", "case_id", "method_number"]:
        df[col] = df[col].astype("int32")
    log.debug("Optimised dtypes.")

    # ------------------------------------------------------------------ #
    # Check missing values before cleaning
    # ------------------------------------------------------------------ #
    log.info("\nMissing values BEFORE cleaning:\n%s", df.isna().sum())

    # ------------------------------------------------------------------ #
    # Fix known data issues
    # ------------------------------------------------------------------ #
    before_fix = df["test_abbr"].isna().sum()
    df.loc[df["test_name"] == "Natrium", "test_abbr"] = "Na"
    log.debug("Fixed Na abbreviation ( %d → %d NaNs left )",
              before_fix, df["test_abbr"].isna().sum())

    # ------------------------------------------------------------------ #
    # Drop rows w/ no result
    # ------------------------------------------------------------------ #
    initial_rows = len(df)
    df.dropna(subset=["text_result", "numeric_result"], how="all", inplace=True)
    log.debug("Dropped %d rows with both results missing.",
             initial_rows - len(df))

    # ------------------------------------------------------------------ #
    # Fill numeric_result from text_result
    # ------------------------------------------------------------------ #
    mask = df["numeric_result"].isna()
    filled = pd.to_numeric(df.loc[mask, "text_result"], errors="coerce")
    df.loc[mask, "numeric_result"] = filled
    log.debug("Filled %d numeric_result values from text_result.", len(filled))

    # ------------------------------------------------------------------ #
    # Final drop of still-missing numeric_result
    # ------------------------------------------------------------------ #
    df.dropna(subset=["numeric_result"], inplace=True)
    log.debug("Remaining rows: %d", len(df))

    # ------------------------------------------------------------------ #
    # Handle duplicates
    # ------------------------------------------------------------------ #
    df = utils.remove_duplicates(df)

    # ------------------------------------------------------------------ #
    # Check missing values after cleaning
    # ------------------------------------------------------------------ #
    log.info("\nMissing values AFTER cleaning:\n%s", df.isna().sum())
    log.info("=== Finished lab data cleaning ===\n")

    # ------------------------------------------------------------------ #
    # Save & finish
    # ------------------------------------------------------------------ #
    if save:
        utils.save_csv(df, config.CLEANED_LAB_PATH)

    return df

def clean_clinical_data(df: pd.DataFrame, save=True, verbose=False) -> pd.DataFrame:
    """
    Cleans clinical data for analysis.

    Steps:
    - Rename German columns to English.
    - Check and handle missing values.
    - Remove duplicate rows.
    - Optionally save the cleaned data.

    Args:
        df (pd.DataFrame): Raw clinical data to be cleaned.
        save (bool): Whether to save the cleaned file. Default True.
        verbose (bool): Whether to log in DEBUG mode. Default False.

    Returns:
        pd.DataFrame: Cleaned clinical DataFrame.
    """

    log.info("\n=== Starting clinical data cleaning ===\n")
    log.debug("Shape after load: %s", df.shape)
    log.debug(f"Number of unique entries per column:\n{df.nunique()}")

    # ------------------------------------------------------------------ #
    # Translate column names and 'discharge' entries from German to English
    # ------------------------------------------------------------------ #    
    df.rename(columns=config.COLUMN_TRANSLATION_CLIN, inplace=True)
    log.debug("Renamed columns to: %s", list(df.columns))

    df['discharge_type'] = df['discharge_type'].map(config.DISCHARGE_TRANSLATION_CLIN)
    log.debug("Translated 'discharge_type' entries to: %s", df['discharge_type'].unique())

    # ------------------------------------------------------------------ #
    # Check missing values before cleaning
    # ------------------------------------------------------------------ #
    log.info("\nMissing values BEFORE cleaning:\n%s", df.isna().sum())

    # ------------------------------------------------------------------ #
    # Drop rows with missing discharge_type
    # ------------------------------------------------------------------ #
    before_drop = df.shape[0]
    df.dropna(subset=["discharge_type"], inplace=True)
    log.debug("Dropped %d rows with missing discharge_type", before_drop - df.shape[0])

    # ------------------------------------------------------------------ #
    # Handle duplicates
    # ------------------------------------------------------------------ #
    df = utils.remove_duplicates(df)

    # ------------------------------------------------------------------ #
    # Check missing values after cleaning
    # ------------------------------------------------------------------ #
    log.info("\nMissing values AFTER cleaning:\n%s", df.isna().sum())
    log.info("=== Finished clinical data cleaning ===\n")

    # ------------------------------------------------------------------ #
    # Save & finish
    # ------------------------------------------------------------------ #
    if save:
        utils.save_csv(df, config.CLEANED_CLINICAL_PATH)

    return df

if __name__ == "__main__":

    lab_data = utils.load_csv(config.LAB_DATA_PATH)
    cleaned_lab_data = clean_lab_data(lab_data, save=True, verbose=True)

    clin_data = utils.load_csv(config.CLINICAL_DATA_PATH)
    cleaned_clin_data = clean_clinical_data(clin_data, save=True, verbose=True)