# -------------------------------
# Project Configuration File
# -------------------------------

# === Imports ===
from pathlib import Path
from collections import namedtuple
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# === Project Paths ===
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

# === Raw Data Paths ===

# Full datasets (commented out when testing)
# LAB_DATA_PATH = RAW_DIR / "RITM0154633_lab.csv"
# CLINICAL_DATA_PATH = RAW_DIR / "RITM0154633_main.csv"

#------------------------------------------------------------------
# For testing locally with smaller subsets

# Paths to save the smaller subsets
RAW_SUBSETS_DIR = TEST_DIR / "subsets"
LAB_SAMPLE_PATH = TEST_DIR / "subsets/sample_lab.csv"
CLINICAL_SAMPLE_PATH = TEST_DIR / "subsets/sample_main.csv"
# Choose which to use by default:
LAB_DATA_PATH = LAB_SAMPLE_PATH
CLINICAL_DATA_PATH = CLINICAL_SAMPLE_PATH
#------------------------------------------------------------------

# === Preprocessed Data Paths ===
# Paths to save the full preprocessed data (commented out when testing)
# CLEANED_LAB_PATH = INTERIM_DIR / "cleaned_lab_data.csv"
# CLEANED_CLINICAL_PATH = INTERIM_DIR / "cleaned_clinical_data.csv"
# MERGED_DATA_PATH = INTERIM_DIR / "merged_data.csv"
# CLEANED_MERGED_DATA_PATH = INTERIM_DIR / "cleaned_merged_data.csv"
# REFERENCE_TABLE_PATH = INTERIM_DIR / "selected_tests_reference_table.csv"
# LAB_TEST_STATISTICS = INTERIM_DIR / "lab_test_statistics.csv"

#------------------------------------------------------------------
# For testing locally with smaller subsets
INTERIM_SUBSETS_DIR = TEST_DIR / "subsets"

# Preprocessed subset paths for testing
CLEANED_LAB_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "cleaned_sample_lab_data.csv"
CLEANED_CLINICAL_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "cleaned_sample_clinical_data.csv"
MERGED_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "merged_sample_data.csv"
CLEANED_MERGED_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "cleaned_merged_sample_data.csv"
REFERENCE_TABLE_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "selected_tests_reference_table_sample.csv"
LAB_TEST_STATISTICS_SAMPLE_PATH = INTERIM_SUBSETS_DIR / "lab_test_statistics_sample.csv"

# Switch between full or sample
CLEANED_LAB_PATH = CLEANED_LAB_SAMPLE_PATH
CLEANED_CLINICAL_PATH = CLEANED_CLINICAL_SAMPLE_PATH
MERGED_DATA_PATH = MERGED_SAMPLE_PATH
CLEANED_MERGED_DATA_PATH = CLEANED_MERGED_SAMPLE_PATH
REFERENCE_TABLE_PATH = REFERENCE_TABLE_SAMPLE_PATH
LAB_TEST_STATISTICS = LAB_TEST_STATISTICS_SAMPLE_PATH
#------------------------------------------------------------------

# === Modeling Configuration ===
LOS_TARGET = "length_of_stay_days"
DISCHARGE_TARGET = "discharge_type"
RANDOM_SEED = 42
CV_FOLDS = 5
TEST_SIZE = 0.20
LOS_TARGET_THRESHOLDS = [7, 14, 30]

# === Default Model Configuration ===
DEFAULT_MODEL_CONFIG = {
    "random_forest": {
        "classifier": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 8,
                "random_state": RANDOM_SEED,
                "class_weight": "balanced"
            }
        },
        "regressor": {
            "class": RandomForestRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": RANDOM_SEED
            }
        }
    },
    "decision_tree": {
        "classifier": {
            "class": DecisionTreeClassifier,
            "params": {
                "max_depth": 5,
                "random_state": RANDOM_SEED
            }
        },
        "regressor": {
            "class": DecisionTreeRegressor,
            "params": {
                "max_depth": 5,
                "random_state": RANDOM_SEED
            }
        }
    },
    "catboost": {
        "classifier": {
            "class": CatBoostClassifier,
            "params": {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6,
                "eval_metric": "F1",          
                "loss_function": "Logloss",
                "random_seed": RANDOM_SEED,
                "verbose": 100    ,
                "auto_class_weights": "Balanced"  
            }
        },
        "regressor": {
            "class": CatBoostRegressor,
            "params": {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": RANDOM_SEED,
                "verbose": 100
            }
        }
    },
    "lightgbm": {
        "classifier": {
            "class": LGBMClassifier,
            "params": {
                "n_estimators": 1000,          # allow early stopping to determine optimal iteration
                "learning_rate": 0.05,         # slower learning for better generalization
                "max_depth": -1,               # let trees grow until leaves
                "num_leaves": 31,              # default, often reasonable
                "min_child_samples": 20,       # helps prevent overfitting
                "subsample": 0.8,              # bagging (row sampling)
                "colsample_bytree": 0.8,       # feature sampling
                "random_state": RANDOM_SEED,
                "verbosity": -1 ,
                "is_unbalance": True               # cleaner output
            }
        },
        "regressor": {
            "class": LGBMRegressor,
            "params": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": RANDOM_SEED,
                "verbosity": -1
            }
        }
    },
    "xgboost": {
        "classifier": {
            "class": XGBClassifier,
            "params": {
                "n_estimators": 1000,              # large number with early stopping
                "learning_rate": 0.05,             # slower learning for better generalization
                "max_depth": 6,                    # shallower trees help generalize
                "min_child_weight": 1,             # minimum sum of instance weight needed in a child
                "subsample": 0.8,                  # row sampling
                "colsample_bytree": 0.8,           # feature sampling
                "gamma": 0,                        # minimum loss reduction
                "random_state": RANDOM_SEED,
                "use_label_encoder": False,
                "eval_metric": "logloss",          # or 'auc' for binary classification
                "verbosity": 0
            }
        },
        "regressor": {
            "class": XGBRegressor,
            "params": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
                "random_state": RANDOM_SEED,
                "verbosity": 0
            }
        }
    }
}

# === Column translation ===
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
    "Entl. Tarifbere": "Discharge tariff status",
    "Entl.eigner Wun": "Discharge on patient's own request",
    "Wartepat. Entl.": "Discharge of a waiting patient",
    "Ambulanz exInst": "Ambulance to external institution",
    "Rückv n. ambBeh": "Transfer back from another institution",
    "Besuch Krankh.": "Visit to the hospital"
}

# === Data filtering ===
MISSING_THRESHOLD = 80  # % missing lab tests per case allowed
REMOVE_TESTS = [
    "EC3-U", "PH4-U", "BI3-U", "LK3-U", "PROT3", "NITR3", "KETO3", "SPEZ3",
    "FARBE3", "TRUEB3", "UST1", "BA-Ux", "LK-Ux", "KRI-Ux", "ERY-Ux", "PI-Ux", "EART",
    "GLUC3", "URO3", "ENTN1n", "Ben-ID", "TNT_hn"
]

# === Feature/Target config ===
LOS_TARGET = "length_of_stay_days"
DISCHARGE_TARGET = "discharge_type"

DISCHARGE_TARGET_CATEGORIES_4 = [
    "Home", "Another hospital", "Institution", "Deceased"
]

DISCHARGE_TARGET_CATEGORIES_3 = [
    "Home", "Another hospital/institution", "Deceased"
]

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

# === Dataset Configurations ===
# Dataset configuration schema
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

# === LightGBM / CatBoost (can handle NaN + categorical) ===
DATASET_CONFIGS_NAN_CAT_NATIVE = [
    DatasetConfig(
        name="raw_minimal",
        apply_age_binning=False,
        add_missing_flags=False,
        icd_strategy=None,
        impute=False,
        encode=False,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="missing_flags_only",
        apply_age_binning=False,
        add_missing_flags=True,
        icd_strategy=None,
        impute=False,
        encode=False,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="engineered_only_nan",
        apply_age_binning=False,
        add_missing_flags=False,
        icd_strategy=None,
        impute=False,
        encode=False,
        scale=False,
        engineered_features=["anemia", "kidney", "liver"],
        ordinal_cols=["anemia_level", "kidney_function", "liver_fibrosis_risk"],
        ordinal_categories=[
            ["Severe Anemia", "Moderate Anemia", "Mild Anemia", "Normal", "Unknown"],
            ["severe", "moderate", "mild", "normal", "unknown"],
            ["high_risk", "moderate_risk", "no_fibrosis", "unknown"]
        ]
    ),
    DatasetConfig(
        name="engineered_and_flags_nan",
        apply_age_binning=False,
        add_missing_flags=True,
        icd_strategy=None,
        impute=False,
        encode=False,
        scale=False,
        engineered_features=["anemia", "kidney", "liver"],
        ordinal_cols=["anemia_level", "kidney_function", "liver_fibrosis_risk"],
        ordinal_categories=[
            ["Severe Anemia", "Moderate Anemia", "Mild Anemia", "Normal", "Unknown"],
            ["severe", "moderate", "mild", "normal", "unknown"],
            ["high_risk", "moderate_risk", "no_fibrosis", "unknown"]
        ]
    ),
    DatasetConfig(
        name="icd_blocks_flags_nan",
        apply_age_binning=True,
        add_missing_flags=True,
        icd_strategy="blocks",
        impute=False,
        encode=False,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="icd_categories_flags_nan",
        apply_age_binning=True,
        add_missing_flags=True,
        icd_strategy="categories",
        impute=False,
        encode=False,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="age_bin_icd_grouped_flags_nan",
        apply_age_binning=True,
        add_missing_flags=True,
        icd_strategy="categories",
        impute=False,
        encode=False,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="age_bin_icd_blocks_flags_nan",
        apply_age_binning=True,
        add_missing_flags=True,
        icd_strategy="blocks",
        impute=False,
        encode=False,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    )
]

# === XGBoost (can handle NaNs but prefers numerical only) ===
DATASET_CONFIGS_NAN_CAT_ENCODED = [
    DatasetConfig(
        name="encoded_only_nan",
        apply_age_binning=False,
        add_missing_flags=False,
        icd_strategy=None,
        impute=False,
        encode=True,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="imputed_encoded_nan",
        apply_age_binning=False,
        add_missing_flags=True,
        icd_strategy=None,
        impute=True,
        encode=True,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="encoded_scaled_nan",
        apply_age_binning=False,
        add_missing_flags=False,
        icd_strategy=None,
        impute=False,
        encode=True,
        scale=True,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="imputed_encoded_scaled_nan",
        apply_age_binning=False,
        add_missing_flags=True,
        icd_strategy=None,
        impute=True,
        encode=True,
        scale=True,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    )
]

# === Models needing full numeric and no NaNs (LogReg, SVM, kNN) ===
DATASET_CONFIGS_NO_NAN_ENCODED = [
    DatasetConfig(
        name="imputed_encoded_scaled",
        apply_age_binning=True,
        add_missing_flags=True,
        icd_strategy="blocks",
        impute=True,
        encode=True,
        scale=True,
        engineered_features=["anemia", "kidney", "liver"],
        ordinal_cols=["anemia_level", "kidney_function", "liver_fibrosis_risk"],
        ordinal_categories=[
            ["Severe Anemia", "Moderate Anemia", "Mild Anemia", "Normal", "Unknown"],
            ["severe", "moderate", "mild", "normal", "unknown"],
            ["high_risk", "moderate_risk", "no_fibrosis", "unknown"]
        ]
    ),
    DatasetConfig(
        name="imputed_encoded",
        apply_age_binning=False,
        add_missing_flags=False,
        icd_strategy=None,
        impute=True,
        encode=True,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="imputed_encoded_icd_grouped",
        apply_age_binning=True,
        add_missing_flags=True,
        icd_strategy="categories",
        impute=True,
        encode=True,
        scale=False,
        engineered_features=[],
        ordinal_cols=None,
        ordinal_categories=None
    ),
    DatasetConfig(
        name="imputed_encoded_engineered",
        apply_age_binning=False,
        add_missing_flags=False,
        icd_strategy=None,
        impute=True,
        encode=True,
        scale=False,
        engineered_features=["anemia", "kidney"],
        ordinal_cols=["anemia_level", "kidney_function"],
        ordinal_categories=[
            ["Severe Anemia", "Moderate Anemia", "Mild Anemia", "Normal", "Unknown"],
            ["severe", "moderate", "mild", "normal", "unknown"]
        ]
    )
]

# === All dataset configs together (optional master list)
ALL_DATASET_CONFIGS = (
    DATASET_CONFIGS_NAN_CAT_NATIVE +
    DATASET_CONFIGS_NAN_CAT_ENCODED +
    DATASET_CONFIGS_NO_NAN_ENCODED
)



