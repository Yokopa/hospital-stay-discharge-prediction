"""
Project Configuration File for Hospital Stay and Discharge Prediction.

This file defines global paths, modeling parameters, target settings, column mappings,
model classes, filtering rules, and dataset configuration structures used across the pipeline.
"""

# -------------------------------
# Project Configuration File
# -------------------------------

# === Imports ===
from pathlib import Path
from collections import namedtuple

from sklearn.linear_model import LogisticRegression, LinearRegression
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# --- Base Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
LOGS_DIR = REPORTS_DIR / "logs"
TEST_DIR = BASE_DIR / "tests"
RESULTS_DIR = BASE_DIR / "results"

# --- Raw Data Files ---
# Full datasets (commented out when testing)
LAB_DATA_PATH = RAW_DIR / "RITM0154633_lab.csv"
CLINICAL_DATA_PATH = RAW_DIR / "RITM0154633_main.csv"

# --- Preprocessed Data Files ---
# Paths to save the full preprocessed data (commented out when testing)
CLEANED_LAB_PATH = INTERIM_DIR / "cleaned_lab_data.csv"
CLEANED_CLINICAL_PATH = INTERIM_DIR / "cleaned_clinical_data.csv"
MERGED_DATA_PATH = INTERIM_DIR / "merged_data.csv"
CLEANED_MERGED_DATA_PATH = INTERIM_DIR / "cleaned_merged_data.csv"
REFERENCE_TABLE_PATH = INTERIM_DIR / "selected_tests_reference_table.csv"
LAB_TEST_STATISTICS = INTERIM_DIR / "lab_test_statistics.csv"

# ------------------------------------------------------------------
# === Local Testing with Smaller Subsets ===
# To use smaller sample data instead of full datasets:
# 1. Uncomment the 3 lines below (LAB_SAMPLE_PATH, CLINICAL_SAMPLE_PATH, LAB_DATA_PATH/CLINICAL_DATA_PATH).
# 2. Comment out the default LAB_DATA_PATH and CLINICAL_DATA_PATH defined above if necessary.
# ------------------------------------------------------------------

# RAW_SUBSETS_DIR = TEST_DIR / "subsets"
# LAB_SAMPLE_PATH = RAW_SUBSETS_DIR / "sample_lab.csv"
# CLINICAL_SAMPLE_PATH = RAW_SUBSETS_DIR / "sample_main.csv"

# Use these instead of full data paths for faster testing
# LAB_DATA_PATH = LAB_SAMPLE_PATH
# CLINICAL_DATA_PATH = CLINICAL_SAMPLE_PATH

# ------------------------------------------------------------------
# === Preprocessed Subsets for Local Testing ===
# To use preprocessed samples:
# 1. Uncomment the 6 paths below under INTERIM_SUBSETS_DIR.
# 2. Then uncomment the 6 overrides (CLEANED_LAB_PATH, etc.).
# 3. Comment out the full-data equivalents if needed.
# ------------------------------------------------------------------

# INTERIM_SUBSETS_DIR = TEST_DIR / "subsets"

# CLEANED_LAB_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "cleaned_sample_lab_data.csv"
# CLEANED_CLINICAL_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "cleaned_sample_clinical_data.csv"
# MERGED_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "merged_sample_data.csv"
# CLEANED_MERGED_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "cleaned_merged_sample_data.csv"
# REFERENCE_TABLE_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "selected_tests_reference_table_sample.csv"
# LAB_TEST_STATISTICS_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "lab_test_statistics_sample.csv"

# Switch to these sample paths for testing:
# CLEANED_LAB_PATH = CLEANED_LAB_SAMPLE_PATH
# CLEANED_CLINICAL_PATH = CLEANED_CLINICAL_SAMPLE_PATH
# MERGED_DATA_PATH = MERGED_SAMPLE_PATH
# CLEANED_MERGED_DATA_PATH = CLEANED_MERGED_SAMPLE_PATH
# REFERENCE_TABLE_PATH = REFERENCE_TABLE_SAMPLE_PATH
# LAB_TEST_STATISTICS = LAB_TEST_STATISTICS_SAMPLE_PATH

# =============================================================================
# MODELING CONFIGURATION
# =============================================================================

# --- General Modeling ---
LOS_TARGET = "length_of_stay_days"
DISCHARGE_TARGET = "discharge_type"
RANDOM_SEED = 42
CV_FOLDS = 5
TEST_SIZE = 0.20

# --- Target Columns ---
LOS_TARGET = "length_of_stay_days"
DISCHARGE_TARGET = "discharge_type"

# --- LOS Classification ---
LOS_TARGET_THRESHOLDS = [[3], [4], [5], [7], [10], [14], [30]] # change to LOS_THRESHOLDS_TO_COMPARE?????
# LOS_TARGET_THRESHOLDS = [
#     [3],           # binary threshold
#     [7, 14],       # multiclass with two thresholds → 3 classes
#     [3, 7, 10, 14] # multiclass with four thresholds → 5 classes
# ]

LOS_CLASSIFICATION_THRESHOLDS = [4, 7]  # Default thresholds for LOS classification and two-step modeling

LOS_TRANSFORMATION = {
    "method": "none",  # possible options: "none", "cap", "winsorize", "log"
    #"method": "winsorize",
    "cap_value": "99%",  # or int value like 60
    "winsor_limits": (0.05, 0.05),
}

# --- Discharge Categories ---
DISCHARGE_CATEGORIES_NUMBER = 3  # change this to 3 to group into fewer classes

DISCHARGE_TARGET_CATEGORIES = {
    3: ["Home", "Another hospital/institution", "Deceased"],
    4: ["Home", "Another hospital", "Institution", "Deceased"]
}

# --- ICD Mapping ---


ICD10_CATEGORIES = {
    "I": "Infectious diseases",
    "II": "Neoplasms",
    "III": "Blood & immune disorders",
    "IV": "Endocrine & metabolic",
    "V": "Mental disorders",
    "VI": "Nervous system diseases",
    "VII": "Eye diseases",
    "VIII": "Ear diseases",
    "IX": "Circulatory diseases",
    "X": "Respiratory diseases",
    "XI": "Digestive diseases",
    "XII": "Skin diseases",
    "XIII": "Musculoskeletal diseases",
    "XIV": "Genitourinary diseases",
    "XV": "Pregnancy & childbirth",
    "XVI": "Perinatal conditions",
    "XVII": "Congenital disorders",
    "XVIII": "Symptoms & abnormal findings",
    "XIX": "Injury & poisoning",
    "XX": "External causes",
    "XXI": "Health factors & services",
    "XXII": "Special codes"
}

# =============================================================================
# MODEL CLASS DEFINITIONS
# =============================================================================

REGRESSOR_CLASSES = {
    "LGBMRegressor": LGBMRegressor,
    "CatBoostRegressor": CatBoostRegressor,
    "XGBRegressor": XGBRegressor, # (can handle NaNs but prefers numerical only)
    "LinearRegression": LinearRegression,         
}

CLASSIFIER_CLASSES = {
    "LGBMClassifier": LGBMClassifier,
    "CatBoostClassifier": CatBoostClassifier,
    "XGBClassifier": XGBClassifier, # (can handle NaNs but prefers numerical only)
    "LogisticRegression": LogisticRegression,       
}

# =============================================================================
# DATA CLEANING & FILTERING
# =============================================================================

MISSING_THRESHOLD = 80  # % missing lab tests per case allowed

REMOVE_TESTS = [
    "EC3-U", "PH4-U", "BI3-U", "LK3-U", "PROT3", "NITR3", "KETO3", "SPEZ3",
    "FARBE3", "TRUEB3", "UST1", "BA-Ux", "LK-Ux", "KRI-Ux", "ERY-Ux", "PI-Ux", "EART",
    "GLUC3", "URO3", "ENTN1n", "Ben-ID", "TNT_hn"
]

# =============================================================================
# COLUMN TRANSLATIONS
# =============================================================================

COLUMN_TRANSLATION_LAB = {
    "dim_patient_bk_pseudo": "patient_id",
    "dim_fall_bk_pseudo": "case_id",
    "bezeichnung": "test_name",
    "kurzbezeichnung": "test_abbr",
    "methodenummer": "method_number",
    "ergebnis_numerisch": "numeric_result",
    "ergebnis_text": "text_result",
    "unit": "unit"
}

COLUMN_TRANSLATION_CLIN = {
    "dim_patient_bk_pseudo": "patient_id",
    "dim_fall_bk_pseudo": "case_id",
    "entlassungsart": "discharge_type",
    "geschlecht": "sex",
    "alter": "age",
    "liegezeit_tage": "length_of_stay_days",
    "hauptdiagnose": "diagnosis"
}

DISCHARGE_TRANSLATION_CLIN = {
    "Entlassung": "Home",
    "Entl.ext.Instit": "Institution",
    "Entl. in ex.KH": "Another hospital",
    "Verstorben": "Deceased",
    "Entl. Tarifbere": "Tariff status", # Discharge tariff status
    "Entl.eigner Wun": "Patient's request", # Discharge on patient's own request
    "Wartepat. Entl.": "Waiting patient", # Discharge of a waiting patient
    "Ambulanz exInst": "Ambulance transfer", # Ambulance to external institution
    "Rückv n. ambBeh": "Transfer back", # Transfer back from another institution
    "Besuch Krankh.": "Hospital visit" # Visit to the hospital
}

# =============================================================================
# CONFIGURATION FILE PATHS
# =============================================================================

DATASET_CONFIG_PATH = "/home/anna/Desktop/Master_thesis/hospital-stay-discharge-prediction/config_experiment/dataset.yml"
MODEL_CONFIG_PATH = "/home/anna/Desktop/Master_thesis/hospital-stay-discharge-prediction/config_experiment/params.yml"


# =============================================================================
# NAMED TUPLES FOR CONFIGURATION
# =============================================================================

DatasetConfig = namedtuple("DatasetConfig", [
    "name",
    "apply_age_binning",
    "add_missing_flags",
    "icd_strategy",
    "impute",
    "encode",
    "scale",
    "engineered_features",
    "ordinal_cols",
    "ordinal_categories"
])



