import pandas as pd
from pathlib import Path
import sys
import os
# Add parent directory to sys.path to find config.py one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.utils as utils
import src.config as config

# Paths to your full raw data files
LAB_PATH = "/home/anna/Desktop/Master_thesis/hospital-stay-discharge-prediction/data/raw/RITM0154633_lab.csv"
CLINICAL_PATH = "/home/anna/Desktop/Master_thesis/hospital-stay-discharge-prediction/data/raw/RITM0154633_main.csv"

# Paths to save the smaller subsets
LAB_SAMPLE_PATH = "/home/anna/Desktop/Master_thesis/hospital-stay-discharge-prediction/tests/subsets/sample_lab.csv"
CLINICAL_SAMPLE_PATH = "/home/anna/Desktop/Master_thesis/hospital-stay-discharge-prediction/tests/subsets/sample_main.csv"

config.RAW_SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
config.INTERIM_SUBSETS_DIR.mkdir(parents=True, exist_ok=True)

def create_lab_sample(n=100_000):
    # Read first n rows of lab data
    lab_df = pd.read_csv(LAB_PATH, nrows=n)
    lab_df.to_csv(LAB_SAMPLE_PATH, index=False)
    print(f"Saved lab sample with {len(lab_df)} rows to {LAB_SAMPLE_PATH}")

def create_clinical_sample(n=20_000):
    # Read first n rows of clinical data
    clinical_df = pd.read_csv(CLINICAL_PATH, nrows=n)
    clinical_df.to_csv(CLINICAL_SAMPLE_PATH, index=False)
    print(f"Saved clinical sample with {len(clinical_df)} rows to {CLINICAL_SAMPLE_PATH}")

if __name__ == "__main__":
    create_lab_sample()
    create_clinical_sample()
