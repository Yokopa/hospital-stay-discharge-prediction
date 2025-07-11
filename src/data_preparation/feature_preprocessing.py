"""
Feature preprocessing utilities for hospital stay/discharge prediction.

This module provides functions for:
- Adding missing value indicators to numerical columns
- Binning and categorizing age
- Grouping and categorizing ICD-10 diagnosis codes
- Encoding categorical variables (ordinal and one-hot)
- Scaling features
- Imputing missing values using Random Forests
- Patient-level train/test splitting to prevent data leakage
- Target preprocessing for classification and regression (LOS) tasks
- Dataset preparation pipeline according to configuration
- Dataset inspection utilities

Intended for use in the data preparation pipeline prior to model training.
"""

import pandas as pd
import numpy as np
# Enables experimental IterativeImputer (must come before import)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
from src.data_preparation.feature_engineering import classify_anemia_dataset, classify_kidney_function, compute_apri
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import sys
import os
# Add parent directory to sys.path to find config.py one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import utils

import logging
log = logging.getLogger(__name__)

# -------------------------
# Missing data flags
# -------------------------
def add_missing_indicators(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Add binary columns indicating missing values in numerical features.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    if columns:
        num_cols = [col for col in columns if col in num_cols]

    for col in num_cols:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    return df

# -------------------------
# Binning age
# -------------------------
def categorize_age(age_series):
    """
    Categorizes ages into defined bins.

    Args:
        age_series (pd.Series): A pandas Series containing age values (e.g., from a DataFrame column).

    Returns:
        pd.Series: A Series with age category labels as strings.
    """
    bins = [18, 45, 60, 75, 110]
    labels = ["18-44", "45-59", "60-74", "75+"]
    categorized = pd.cut(age_series, bins=bins, labels=labels, right=False)
    return categorized.astype(str).replace('nan', 'unknown')

# -------------------------
# Grouping ICD codes
# -------------------------
def group_icd_to_blocks(df, column='diagnosis'):
    """
    Truncates ICD-10-GM codes in a DataFrame column to their first three characters.

    Args:
        df (pd.DataFrame): The DataFrame containing ICD codes.
        column (str): The name of the column with ICD codes. Default is 'diagnosis'.

    Returns:
        pd.DataFrame: The same DataFrame with the specified column updated.
    """
    df = df.copy()
    df[column] = df[column].astype(str).str[:3]
    return df

def categorize_icd(icd_code : str, icd_categories : dict=None):
    """
    Assigns an ICD-10 code to its main category based on the first character(s).
    
    Args:
        icd_code (str): The ICD-10 diagnosis code (e.g., 'I25.19').
        icd_categories (dict): Optional dictionary of ICD categories. 
                               If None, uses ICD10_CATEGORIES from config.

    Returns:
        str: The main ICD-10 category, or 'Unknown' if not matched.
    
    Example:
        df['icd_category'] = df['diagnosis'].apply(categorize_icd)
    """
    if icd_categories is None:
        icd_categories = config.ICD10_CATEGORIES

    icd_code = str(icd_code).strip().upper()

    if icd_code.startswith(("A", "B")):
        return icd_categories["I"]
    elif icd_code.startswith(("C", "D0", "D1", "D2", "D3", "D4")):
        return icd_categories["II"]  # Neoplasms
    elif icd_code.startswith(("D5", "D6", "D7", "D8")):
        return icd_categories["III"]  # Blood & immune disorders
    elif icd_code.startswith("D9"):
        return icd_categories["XXII"]  # Special codes
    elif icd_code.startswith("E"):
        return icd_categories["IV"]
    elif icd_code.startswith("F"):
        return icd_categories["V"]
    elif icd_code.startswith("G"):
        return icd_categories["VI"]
    elif icd_code.startswith("H"):
        try:
            h_number = int(icd_code[1:3])
            if 0 <= h_number <= 59:
                return icd_categories["VII"]  # Eye diseases
            elif 60 <= h_number <= 95:
                return icd_categories["VIII"]  # Ear diseases
        except (ValueError, IndexError):
            pass
        return "Unknown"
    elif icd_code.startswith("I"):
        return icd_categories["IX"]
    elif icd_code.startswith("J"):
        return icd_categories["X"]
    elif icd_code.startswith("K"):
        return icd_categories["XI"]
    elif icd_code.startswith("L"):
        return icd_categories["XII"]
    elif icd_code.startswith("M"):
        return icd_categories["XIII"]
    elif icd_code.startswith("N"):
        return icd_categories["XIV"]
    elif icd_code.startswith("O"):
        return icd_categories["XV"]
    elif icd_code.startswith("P"):
        return icd_categories["XVI"]
    elif icd_code.startswith("Q"):
        return icd_categories["XVII"]
    elif icd_code.startswith("R"):
        return icd_categories["XVIII"]
    elif icd_code.startswith(("S", "T")):
        return icd_categories["XIX"]
    elif icd_code.startswith(("V", "W", "X", "Y")):
        return icd_categories["XX"]
    elif icd_code.startswith("Z"):
        return icd_categories["XXI"]
    elif icd_code.startswith("U"):
        return icd_categories["XXII"]
    if not icd_code or pd.isna(icd_code):
        return "Unknown"
    else:
        return "Unknown"

# -------------------------
# Encoding categorical variables
# -------------------------
def fit_encoders(X_train, ordinal_cols=None, ordinal_categories=None, drop_first=True):
    """
    Fits OrdinalEncoder and OneHotEncoder to categorical columns in the training dataset.
    
    Parameters:
        X_train (pd.DataFrame): The training DataFrame.
        ordinal_cols (list): List of column names that should be treated as ordinal.
        ordinal_categories (dict): Optional dict mapping ordinal column names to their ordered categories.
        drop_first (bool): Whether to drop the first category in one-hot encoding (to avoid multicollinearity).
    
    Returns:
        dict: A dictionary containing fitted encoders and the associated column names.
    
    Example:
        ordinal_cols = ['education_level']
        ordinal_categories = {'education_level': ['low', 'medium', 'high']}
        
        encoders = fit_encoders(X_train, ordinal_cols, ordinal_categories)
    """
    encoders = {}

    if ordinal_cols is None:
        ordinal_cols = []
    if ordinal_categories is None:
        ordinal_categories = {}

    all_cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    nominal_cols = [col for col in all_cat_cols if col not in ordinal_cols]

    # Fit OrdinalEncoder
    if ordinal_cols:
        categories_list = [ordinal_categories.get(col, None) for col in ordinal_cols]
        ord_encoder = OrdinalEncoder(categories=categories_list)
        ord_encoder.fit(X_train[ordinal_cols])
        encoders['ordinal'] = (ord_encoder, ordinal_cols)

    # Fit OneHotEncoder
    if nominal_cols:
        ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None, handle_unknown='ignore')
        ohe.fit(X_train[nominal_cols])
        encoders['onehot'] = (ohe, nominal_cols)

    return encoders


def transform_with_encoders(X, encoders):
    """
    Transforms a DataFrame using previously fitted encoders.
    
    Parameters:
        X (pd.DataFrame): The input DataFrame (train or test).
        encoders (dict): Dictionary containing fitted encoders from `fit_encoders()`.
    
    Returns:
        pd.DataFrame: Transformed DataFrame with encoded categorical features.
    
    Example:
        X_train_encoded = transform_with_encoders(X_train, encoders)
        X_test_encoded = transform_with_encoders(X_test, encoders)
    """
    X = X.copy()

    # Apply Ordinal Encoding
    if 'ordinal' in encoders:
        ord_encoder, ordinal_cols = encoders['ordinal']
        X[ordinal_cols] = ord_encoder.transform(X[ordinal_cols])

    # Apply One-Hot Encoding
    if 'onehot' in encoders:
        ohe, nominal_cols = encoders['onehot']
        ohe_encoded = ohe.transform(X[nominal_cols])
        ohe_cols = ohe.get_feature_names_out(nominal_cols)
        X_encoded = pd.DataFrame(ohe_encoded, columns=ohe_cols, index=X.index)

        X = X.drop(columns=nominal_cols)
        X = pd.concat([X, X_encoded], axis=1)

    return X

# Example:
# ordinal_cols = ['education']
# ordinal_categories = {'education': ['primary', 'secondary', 'tertiary']}
# encoders = fit_encoders(X_train, ordinal_cols=ordinal_cols, ordinal_categories=ordinal_categories)
# X_train_enc = transform_with_encoders(X_train, encoders)
# X_test_enc = transform_with_encoders(X_test, encoders)

# -------------------------
# Feature scaling
# -------------------------
def scale_features(X_train, X_test, return_scaler=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    if return_scaler:
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, X_test_scaled

# -------------------------
# Imputation with Random Forest
# -------------------------
def impute_with_random_forest(train_df, test_df, impute_cfg=None, random_state=config.RANDOM_SEED):
    """
    Impute missing numerical values in train and test DataFrames using
    IterativeImputer with RandomForestRegressor, using parameters from config or dataset YAML.

    Parameters:
        train_df (pd.DataFrame): Training set (numerical features only)
        test_df (pd.DataFrame): Test set (same columns as train_df)
        impute_cfg (dict or None): Dict with keys like 'n_estimators' and 'max_iter'
        random_state (int): Random seed for reproducibility

    Returns:
        imputed_train_df (pd.DataFrame), imputed_test_df (pd.DataFrame)
    """
    # Validate columns
    assert list(train_df.columns) == list(test_df.columns), "Train and test must have the same columns"

    # Defaults if no config passed
    impute_cfg = impute_cfg or {}
    n_estimators = impute_cfg.get("n_estimators", 5)
    max_iter = impute_cfg.get("max_iter", 3)

    log.info(f"Imputing with RandomForest: n_estimators={n_estimators}, max_iter={max_iter}")

    # Convert to NumPy for sklearn compatibility
    train_array = train_df.to_numpy()
    test_array = test_df.to_numpy()

    # Initialize the imputer
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        max_iter=max_iter,
        random_state=random_state
    )

    # Fit and transform
    log.info("Fitting imputer on training data...")
    imputed_train = imputer.fit_transform(train_array)
    log.info("Transforming test data...")
    imputed_test = imputer.transform(test_array)

    # Convert back to DataFrame
    imputed_train_df = pd.DataFrame(imputed_train, columns=train_df.columns, index=train_df.index)
    imputed_test_df = pd.DataFrame(imputed_test, columns=test_df.columns, index=test_df.index)

    log.info("Imputation completed successfully.")
    return imputed_train_df, imputed_test_df

def clip_imputed_values(original_df, imputed_df):
    """
    Corrects out-of-range imputed values by clipping them to the min/max
    of observed (non-missing) values in the original dataset.
    
    Parameters:
        original_df (pd.DataFrame): The original dataset before imputation (with NaNs)
        imputed_df (pd.DataFrame): The imputed dataset (no NaNs)

    Returns:
        pd.DataFrame: Corrected imputed dataset
    """
    corrected_df = imputed_df.copy()

    for col in original_df.columns:
        if pd.api.types.is_numeric_dtype(original_df[col]):
            valid_values = original_df[col].dropna()
            min_val = valid_values.min()
            max_val = valid_values.max()
            
            # Clip values outside the original observed range
            corrected_df[col] = corrected_df[col].clip(lower=min_val, upper=max_val)
    
    return corrected_df

# Example:
# imputed_train, imputed_test = impute_with_random_forest(train_num_df, test_num_df)
# imputed_train = clip_imputed_values(train_num_df, imputed_train)
# imputed_test = clip_imputed_values(test_num_df, imputed_test)

# -------------------------
# Split a DataFrame into train/test sets by patient_id to prevent data leakage
# -------------------------
def train_test_split_by_patient(
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.20,
        random_state: int = config.RANDOM_SEED,
        stratify: bool = True
    ):
    """
    Split a dataframe into train/test ensuring that each patient_id
    appears in only one set (no leakage).

    Args:
        df : pd.DataFrame
            Your merged dataset containing `patient_id`, features and target.
        target_col : str
            Column name of the prediction target (e.g. 'length_of_stay_days'
            for regression or 'discharge_type' for classification).
        test_size : float
            Fraction of patients to allocate to the test set.
        random_state : int
            Reproducible RNG seed.
        stratify : bool
            If True and the target is categorical, preserves class proportions 
            between train and test using a simple label-proportional heuristic.

    Returns:
        X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """

    # Separate features / target and define the grouping variable
    y = df[target_col]
    X = df.drop(columns=[target_col])
    groups = df['patient_id']

    # Prepare a group splitter
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    # Optionally create coarse strata to keep label proportions
    # (only meaningful for a classification target)
    if stratify and y.nunique() < 30 and y.dtype.name != 'float':
        patient_label = (
            df[['patient_id', target_col]]
            .drop_duplicates('patient_id')
            .set_index('patient_id')[target_col]
        )
        strata = groups.map(patient_label)
        split_gen = splitter.split(X, strata, groups)
    else:
        split_gen = splitter.split(X, groups=groups)

    train_idx, test_idx = next(split_gen)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Sanity check – assert no overlap in patient_id
    assert set(X_train['patient_id']).isdisjoint(X_test['patient_id']), \
        "Data leakage detected: overlapping patient_id in train/test!"

    # Drop id columns AND other target-like variables
    drop_cols = ['patient_id', 'case_id']
    if target_col != 'discharge_type' and 'discharge_type' in X.columns:
        drop_cols.append('discharge_type')
    if target_col != 'length_of_stay_days' and 'length_of_stay_days' in X.columns:
        drop_cols.append('length_of_stay_days')

    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    return X_train, X_test, y_train, y_test

# -------------------------
# Target preprocessing utilities for classification tasks.
# -------------------------
def map_discharge_type(val, target_classes=config.DISCHARGE_CATEGORIES_NUMBER):
    if target_classes == 3:
        if val in ["Another hospital", "Institution"]:
            return "Another hospital/institution"
    return val

def encode_target(y_train, y_test):
    """
    Encodes target labels using LabelEncoder.
    
    Args:
        y_train (array-like): Training target labels.
        y_test (array-like): Test target labels.
        
    Returns:
        y_train_encoded (np.ndarray): Encoded training labels.
        y_test_encoded (np.ndarray): Encoded test labels.
        encoder (LabelEncoder): Fitted encoder (for decoding later).
    """
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    print("LabelEncoder classes and their corresponding encoded labels:")
    for idx, label in enumerate(encoder.classes_):
        print(f"  {label} --> {idx}")
    
    return y_train_encoded, y_test_encoded, encoder

def decode_target(y_pred, encoder):
    """
    Decodes label-encoded predictions back to original labels.
    
    Args:
        y_pred (array-like): Encoded predictions.
        encoder (LabelEncoder): Fitted encoder used for original encoding.
    
    Returns:
        np.ndarray: Decoded labels.
    """
    return encoder.inverse_transform(y_pred)

# Example:
# y_train_enc, y_test_enc, label_enc = encode_target(y_train, y_test)
# Train model...
# y_pred = model.predict(X_test)
# Decode predictions:
# y_pred_labels = decode_target(y_pred, label_enc)

# -------------------------
# Target preprocessing utilities for los
# -------------------------
def transform_los_target(y_train, y_test, los_transform_config):
    log.info("Starting LOS target transformation...")
    method = los_transform_config.get("method", "none")
    log.info(f"Transformation method: {method}")

    if method == "none":
        log.info("No LOS target transformation applied.")
        return y_train, y_test
    
    elif method == "cap":
        cap_val = los_transform_config.get("cap_value", None)

        if cap_val is None:
            cap_val = y_train.quantile(0.99)
            log.info(f"No cap_value specified in config. Using default 99th percentile: {cap_val:.4f}")

        elif isinstance(cap_val, str) and cap_val.endswith('%'):
            try:
                percentile = float(cap_val.strip('%')) / 100
                cap_val = y_train.quantile(percentile)
                log.info(f"Cap value set to {cap_val:.4f} from percentile {percentile*100:.1f}%")
            except ValueError:
                raise ValueError(f"Invalid percentile format in cap_value: {cap_val}")

        elif isinstance(cap_val, (int, float)):
            log.info(f"Cap value set to fixed value: {cap_val}")

        else:
            raise ValueError(f"Invalid cap_value type: {cap_val} (must be float, int, or 'NN%')")

        y_train_capped = y_train.clip(upper=cap_val)
        y_test_capped = y_test.clip(upper=cap_val)
        log.info("LOS target capped successfully.")
        return y_train_capped, y_test_capped

    elif method == "winsorize":
        limits = los_transform_config.get("winsor_limits", (0, 0.01))
        lower_lim = y_train.quantile(limits[0])
        upper_lim = y_train.quantile(1 - limits[1])
        log.info(f"Winsorizing with limits: lower={lower_lim}, upper={upper_lim}")

        y_train_wins = y_train.clip(lower=lower_lim, upper=upper_lim)
        y_test_wins = y_test.clip(lower=lower_lim, upper=upper_lim)
        log.info("LOS target winsorized successfully.")
        return y_train_wins, y_test_wins

    elif method == "log":
        log.info("Applying log1p transformation to LOS target.")
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        log.info("Log transformation completed.")
        return y_train_log, y_test_log

    else:
        raise ValueError(f"Unknown LOS transformation method: {method}")


# -------------------------
# Prepare dataset according to configuration
# -------------------------
def prepare_dataset(raw_df, target_col, dataset_config, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED):
    """
    Prepare dataset according to DatasetConfig with grouped patient split.

    Returns:
        X_train, X_test, y_train, y_test, transformers_dict
    """
    log.info("Starting dataset preparation...\n")
    df = raw_df.copy()

    # ---------------- Engineered features ----------------
    for feat in dataset_config.get("engineered_features", []):
        log.info(f"Applying engineered feature: {feat}")
        if feat == "anemia":
            df = classify_anemia_dataset(df)
            log.info("Anemia classification added.")

        elif feat == "kidney":
            df = classify_kidney_function(df)
            log.info("Kidney function classification added.")

        elif feat == "liver":
            df = compute_apri(df)
            log.info("Liver APRI score added.")

        log.info("Engineered features applied.")

    # ---------------- Deterministic transformations ----------------
    if dataset_config.get("add_missing_flags", False):
        df = add_missing_indicators(df)
        log.info("Missing value indicators added.")

    if dataset_config.get("apply_age_binning", False):
        df["age"] = categorize_age(df["age"])
        log.info("Age binning applied.")

    icd_strategy = dataset_config.get("icd_strategy", None)
    if icd_strategy == "blocks":
        df = group_icd_to_blocks(df)
        log.info("ICD codes grouped to blocks.")
    elif icd_strategy == "categories":
        df["diagnosis"] = df["diagnosis"].apply(categorize_icd)
        log.info("ICD codes categorized.")

    # ---------------- Convert object columns to category ----------------
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype("category")
    log.info(f"Converted object columns to category: {list(obj_cols)}")

    # ---------------- Map & filter discharge_type target (if applicable) ----------------
    if target_col == "discharge_type":
        log.info(" ")
        discharge_classes = config.DISCHARGE_CATEGORIES_NUMBER
        log.info(f"Mapping discharge_type target with {discharge_classes} classes.")
        # Apply mapping function
        df[target_col] = df[target_col].apply(lambda val: map_discharge_type(val, discharge_classes))

        # Filter to allowed classes
        allowed_classes = config.DISCHARGE_TARGET_CATEGORIES[discharge_classes]
        df = df[df[target_col].isin(allowed_classes)].copy()
        log.info(f"Filtered to allowed discharge classes: {sorted(df[target_col].unique())}")
        
        # Add this line to remove unused categories from the target variable
        df[target_col] = df[target_col].astype('category')
        df[target_col] = df[target_col].cat.remove_unused_categories()

        log.info(f"Removed unused categories from {target_col}. Final categories: {list(df[target_col].cat.categories)}")

    # ---------------- Split by patient to avoid leakage ----------------
    log.info(" ")  # blank line
    log.info("Starting train/test split by patient to avoid leakage...")
    X_train, X_test, y_train, y_test = train_test_split_by_patient(
        df, 
        target_col,
        test_size=test_size,
        random_state=random_state,
        stratify=True
    )
    log.info(f"Split done. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Check unseen categories in test vs train
    cat_cols = X_train.select_dtypes(include="category").columns
    for col in cat_cols:
        train_vals = set(X_train[col].dropna().unique())
        test_vals = set(X_test[col].dropna().unique())

        unseen_in_test = test_vals - train_vals
        unseen_in_train = train_vals - test_vals

        if unseen_in_test:
            log.warning(f"[{col}] Categories in test but not in train: {sorted(unseen_in_test)}")
        if unseen_in_train:
            log.info(f"[{col}] Categories in train but not in test: {sorted(unseen_in_train)}")

    log.info("Finished checking category mismatches.\n")

    # ---------------- Apply LOS target transformation (if applicable) ----------------
    # AFTER train/test split to ensure no data leakage by fitting transformation only on train set
    if target_col == config.LOS_TARGET:
        log.info("Transforming LOS target as per config...")
        y_train, y_test = transform_los_target(y_train, y_test, config.LOS_TRANSFORMATION)
        log.info("LOS target transformation completed.\n")

    transformers = {}

    # ---------------- Identify numeric columns ----------------
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    # ---------------- Encoding categorical variables ----------------
    if dataset_config.get("encode", False):
        log.info("Starting encoding of categorical variables...")
        ordinal_cols = dataset_config.get("ordinal_cols", [])
        ordinal_categories = dataset_config.get("ordinal_categories", {})
        
        encoders = fit_encoders(
            X_train,
            ordinal_cols=ordinal_cols,
            ordinal_categories=ordinal_categories
        )
        X_train = transform_with_encoders(X_train, encoders)
        X_test = transform_with_encoders(X_test, encoders)
        transformers['encoders'] = encoders
        log.info("Encoding completed.\n")

    # ---------------- Imputation of numeric columns ----------------
    if dataset_config.get("impute", False):
        impute_cfg = dataset_config.get("imputation_params", {})
        impute_method = dataset_config.get("imputation_method", "random_forest")
        log.info(f"Imputation method selected: {impute_method}")

        if impute_method == "random_forest":
            X_train_num_imp, X_test_num_imp = impute_with_random_forest(
                X_train[numeric_cols],
                X_test[numeric_cols],
                impute_cfg=impute_cfg
            )
            transformers["imputer"] = f"RandomForestImputer(n_estimators={impute_cfg.get('n_estimators', 10)}, max_iter={impute_cfg.get('max_iter', 5)})"

        elif impute_method == "median":
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            X_train_num_imp = pd.DataFrame(
                imputer.fit_transform(X_train[numeric_cols]),
                columns=numeric_cols,
                index=X_train.index
            )
            X_test_num_imp = pd.DataFrame(
                imputer.transform(X_test[numeric_cols]),
                columns=numeric_cols,
                index=X_test.index
            )
            transformers["imputer"] = "SimpleImputer(strategy='median')"

        else:
            raise ValueError(f"Unknown imputation method: {impute_method}")

        # (Negative check and capping - shared for both imputation types)
        log.info("Checking for negative values BEFORE capping in X_train")
        neg_cols_train = utils.find_negative_columns(X_train[numeric_cols])
        log.info(f"Negative columns BEFORE capping (train): {neg_cols_train}")
        log.info("Checking for negative values BEFORE capping in X_test")
        neg_cols_test = utils.find_negative_columns(X_test[numeric_cols])
        log.info(f"Negative columns BEFORE capping (train): {neg_cols_train}")

        X_train.loc[:, numeric_cols] = clip_imputed_values(X_train[numeric_cols], X_train_num_imp)
        X_test.loc[:, numeric_cols] = clip_imputed_values(X_test[numeric_cols], X_test_num_imp)

        log.info("Checking for negative values AFTER capping in X_train")
        utils.find_negative_columns(X_train[numeric_cols])
        log.info("Checking for negative values AFTER capping in X_test")
        utils.find_negative_columns(X_test[numeric_cols])

        log.info("Imputation completed and applied to dataset.\n")

        
    # ---------------- Scaling features ----------------
    if dataset_config.get("scale", False):
        log.info("Starting scaling of features...")
        X_train, X_test, scaler = scale_features(X_train, X_test, return_scaler=True)
        transformers['scaler'] = scaler
        log.info("Scaling completed.\n")

    log.info("Dataset preparation finished successfully.\n")

    # ---------------- Final Sanity Checks & log ----------------
    log.info("-" * 60)
    log.info("Final Sanity Checks & Data Summary\n")

    # --- y_train / y_test summary
    log.debug("y_train distribution:\n%s", y_train.describe())
    log.debug("y_test distribution:\n%s\n", y_test.describe())

    # --- Class balance (only if classification)
    if y_train.dtype.name == 'category' or y_train.dtype == object or y_train.nunique() <= 20:
        log.info("y_train value counts:\n%s", y_train.value_counts())
        log.info("y_test value counts:\n%s\n", y_test.value_counts())

    # --- Feature matrix info
    log.info(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}\n")

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train is not a pandas DataFrame at the end of prepare_dataset!")

    for split_name, X in [("Train", X_train), ("Test", X_test)]:
        # 1) Are there still any categorical columns?
        cat_cols = X.select_dtypes(include="category").columns.tolist()
        if cat_cols:
            log.warning(f"{split_name}: Found unencoded categorical columns: {cat_cols}")

        # 2) Are all the rest numeric?
        num_cols = X.select_dtypes(include="number").columns.tolist()
        if len(num_cols) != X.shape[1]:
            log.warning(f"{split_name}: Some features are non-numeric: " 
                        f"{set(X.columns) - set(num_cols)}")

        # 3) Check for constant columns (useless for ML)
        constant = [c for c in X.columns if X[c].nunique() <= 1]
        if constant:
            log.warning(f"{split_name}: Constant columns: {constant}")

        # 4) If you scaled, verify the means/stds
        if "scale" in dataset_config and dataset_config["scale"]:
            mean = X[num_cols].mean().mean()
            std  = X[num_cols].std().mean()
            log.info(f"{split_name} after scaling: mean≈{mean:.3f}, std≈{std:.3f}\n")
            
    log.info("All sanity checks completed.")
    log.info("-" * 60)

    return X_train, X_test, y_train, y_test, transformers

# -------------------------
# Print dataset information
# -------------------------
def inspect_dataset(config_name, data):
    """
    Print detailed dataset information including shapes, data types, memory usage,
    sample rows, and transformers used.

    Parameters:
    - config_name (str): Name of the dataset configuration.
    - data (dict): Dictionary with keys: 'X_train', 'X_test', 'y_train', 'y_test', 'transformers'.
    """
    print(f"\n=== Dataset: {config_name} ===")

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    transformers = data.get("transformers", {})

    # Shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    # Info and memory usage for features
    print("\nX_train.info():")
    print("-" * 20)
    X_train.info()

    print("\nX_test.info():")
    print("-" * 20)
    X_test.info()

    # Info for targets
    print("\ny_train.info():")
    print("-" * 20)
    if hasattr(y_train, "info"):  # Pandas Series or DataFrame
        y_train.info()
    else:
        print(f"Type: {type(y_train)} — Cannot call .info()")

    print("\ny_test.info():")
    print("-" * 20)
    if hasattr(y_test, "info"):
        y_test.info()
    else:
        print(f"Type: {type(y_test)} — Cannot call .info()")

    # Sample data
    print("\nX_train head():")
    print("-" * 20)
    print(X_train.head())

    print("\nX_test head():")
    print("-" * 20)
    print(X_test.head())

    print("\ny_train head():")
    print("-" * 20)
    print(y_train.head())

    print("\ny_test head():")
    print("-" * 20)
    print(y_test.head())

    # Transformers used
    print("\nTransformers used:")
    if not transformers:
        print("  (None)")
    else:
        for name, transformer in transformers.items():
            if isinstance(transformer, str):
                print(f"  - {name}: {transformer}")
            else:
                print(f"  - {name}: {type(transformer).__name__}")

    print("-" * 60)