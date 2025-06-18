import os
import config
import utils
from data_preparation import clean_raw_data, integrate_data
import pandas as pd
import logging

def load_or_clean_data(cleaned_path):
    if os.path.exists(cleaned_path):
        logging.info(f"Loading cleaned merged data from {cleaned_path}")
        return pd.read_csv(cleaned_path)
    else:
        logging.info("Cleaned data not found. Running full data cleaning pipeline...")

        # Load raw data
        lab_data = utils.load_csv(config.LAB_DATA_PATH)
        clin_data = utils.load_csv(config.CLINICAL_DATA_PATH)

        # Clean raw data
        lab_data = clean_raw_data.clean_lab_data(lab_data)
        clin_data = clean_raw_data.clean_clinical_data(clin_data)

        # Merge and clean
        merged = integrate_data.integrate_data(clin_data, lab_data)
        cleaned = integrate_data.clean_merged_data(merged, save=False)

        # Save result
        cleaned.to_csv(cleaned_path, index=False)
        logging.info(f"Saved cleaned merged data to {cleaned_path}")

        return cleaned