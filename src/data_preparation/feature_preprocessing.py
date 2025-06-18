"""
Data preprocessing utilities for hospital stay/discharge prediction.

Includes:
- Binning (age)
- ICD code grouping/categorization
- Missing value flagging
- Categorical encoding (ordinal, one-hot)
- Feature scaling
- Imputation (Random Forest)
- Patient-level train/test splitting (no leakage)
- Target preprocessing/encoding
- Dataset preparation pipeline
"""

import pandas as pd
import numpy as np
from src import config
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
import logging
import utils
utils.configure_logging(verbose=True)
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
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# -------------------------
# Imputation with Random Forest
# -------------------------
def impute_with_random_forest(train_df, test_df, random_state=config.RANDOM_SEED):
    """
    Impute missing numerical values in train and test DataFrames using
    IterativeImputer with RandomForestRegressor.
    
    Parameters:
        train_df (pd.DataFrame): Training set (numerical features only)
        test_df (pd.DataFrame): Test set (same columns as train_df)
        random_state (int): Random seed for reproducibility
    
    Returns:
        imputed_train_df, imputed_test_df (pd.DataFrame, pd.DataFrame)
    """
    # Check for matching columns
    assert list(train_df.columns) == list(test_df.columns), "Train and test must have the same columns"

    # Convert to numpy arrays
    train_array = train_df.to_numpy()
    test_array = test_df.to_numpy()

    # Initialize IterativeImputer with RandomForest
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=random_state),
        max_iter=20,
        random_state=random_state
    )

    # Fit on training data and transform both
    imputed_train = imputer.fit_transform(train_array)
    imputed_test = imputer.transform(test_array)

    # Convert back to DataFrames
    imputed_train_df = pd.DataFrame(imputed_train, columns=train_df.columns, index=train_df.index)
    imputed_test_df = pd.DataFrame(imputed_test, columns=test_df.columns, index=test_df.index)

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
def discharge_type_preprocess(
    df: pd.DataFrame,
    target_col: str = "discharge_type",
    mode: str = "4_categories",
    allowed_categories: List[str] = None
) -> pd.DataFrame:
    """
    Preprocess and optionally filter the discharge_type target variable.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the discharge type column.
        mode (str): '4_categories' (default) or '3_categories'.
                    - '3_categories' merges 'Discharge to another hospital' and 
                      'Discharge to another institution' into a single category.
        allowed_categories (List[str], optional): List of categories to retain after processing.
                                                  If None, no filtering is applied.

    Returns:
        pd.DataFrame: DataFrame with processed and optionally filtered discharge_type.
    """
    df = df.copy()

    if mode == "4_categories":
        # No change to original values
        df[target_col] = df[target_col].fillna("Unknown")

    elif mode == "3_categories":
        mapping = {
            "Home": "Home",
            "Another hospital": "Another hospital/institution",
            "Institution": "Another hospital/institution",
            "Deceased": "Deceased"
        }
        df[target_col] = df[target_col].map(mapping).fillna("Unknown")

    else:
        raise ValueError("Unsupported mode. Choose either '4_categories' or '3_categories'.")

    if allowed_categories is not None:
        df = df[df[target_col].isin(allowed_categories)]

    return df

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
# Prepare dataset according to configuration
# -------------------------
def prepare_dataset(raw_df, target_col, dataset_config, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED):
    """
    Prepare dataset according to DatasetConfig with grouped patient split.

    Returns:
        X_train, X_test, y_train, y_test, transformers_dict
    """
    df = raw_df.copy()

    # Add engineered features
    for feat in dataset_config.engineered_features:
        if feat == "anemia":
            df = classify_anemia_dataset(df)
        elif feat == "kidney":
            df = classify_kidney_function(df)
        elif feat == "liver":
            df = compute_apri(df)
        # add more as needed

    # Apply deterministic transformations
    if dataset_config.add_missing_flags:
        df = add_missing_indicators(df)

    if dataset_config.apply_age_binning:
        df["age"] = categorize_age(df["age"])

    if dataset_config.icd_strategy == "blocks":
        df = group_icd_to_blocks(df)
    elif dataset_config.icd_strategy == "categories":
        df["diagnosis"] = df["diagnosis"].apply(categorize_icd)

    # Convert object columns to category
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].astype("category")

    # Split by patient_id to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split_by_patient(
        df, 
        target_col,
        test_size=test_size,
        random_state=random_state,
        stratify=True
    )

    transformers = {}

    # Encoding (fit on train only, transform both)
    if dataset_config.encode:
        encoders = fit_encoders(
            X_train,
            ordinal_cols=dataset_config.ordinal_cols,
            ordinal_categories=dataset_config.ordinal_categories
        )
        X_train = transform_with_encoders(X_train, encoders)
        X_test = transform_with_encoders(X_test, encoders)
        transformers['encoders'] = encoders

    # Imputation on numeric columns (fit on train, apply on test)
    if dataset_config.impute:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_num_imp, X_test_num_imp = impute_with_random_forest(
            X_train[numeric_cols], X_test[numeric_cols]
        )
        X_train_num_imp = clip_imputed_values(X_train[numeric_cols], X_train_num_imp)
        X_test_num_imp = clip_imputed_values(X_test[numeric_cols], X_test_num_imp)

        X_train.loc[:, numeric_cols] = X_train_num_imp
        X_test.loc[:, numeric_cols] = X_test_num_imp
        transformers['imputer'] = "RandomForestImputer"  # or store fitted imputer model if desired

    # Scaling (fit scaler on train, apply on test)
    if dataset_config.scale:
        X_train, X_test, scaler = scale_features(X_train, X_test, return_scaler=True)
        transformers['scaler'] = scaler

    return X_train, X_test, y_train, y_test, transformers

# # -------------------------
# # Generate multiple datasets from configurations
# # -------------------------
# def generate_all_datasets(raw_df, target_col, dataset_configs):
#     """
#     Generates datasets for each DatasetConfig.
#     Returns dict: {config.name: (X_train, X_test, y_train, y_test, transformers)}
#     """
#     all_data = {}
#     for config in dataset_configs:
#         log.info(f"Preparing dataset: {config.name}")
#         X_train, X_test, y_train, y_test, transformers = prepare_dataset(raw_df, target_col, config)
#         all_data[config.name] = {
#             "X_train": X_train,
#             "X_test": X_test,
#             "y_train": y_train,
#             "y_test": y_test,
#             "transformers": transformers,
#             "config": config,
#         }
#     return all_data

# Example usage:

# datasets_lgbm = generate_all_datasets(raw_df, DATASET_CONFIGS_NAN_CAT_NATIVE)
# datasets_xgb = generate_all_datasets(raw_df, DATASET_CONFIGS_NAN_CAT_ENCODED)
# datasets_classical = generate_all_datasets(raw_df, DATASET_CONFIGS_NO_NAN_ENCODED)

# Or all combined:
# all_datasets = generate_all_datasets(raw_df, ALL_DATASET_CONFIGS)

# Access dataset:
# X_train = all_datasets['imputed_encoded_scaled']['X_train']
# y_train = all_datasets['imputed_encoded_scaled']['y_train']

# -------------------------
# Print dataset information
# -------------------------
def inspect_dataset(config_name, data):
    """
    Print dataset information including shape, sample data, and transformers used.

    Parameters:
    - config_name (str): Name of the dataset configuration.
    - data (dict): Dictionary with keys: 'X_train', 'X_test', 'y_train', 'y_test', 'transformers'.
    """
    print(f"\n=== Dataset: {config_name} ===")

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    transformers = data["transformers"]

    # Dataset shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    # Data type and memory info
    print("\nX_train.info():")
    print("-" * 20)
    X_train.info()

    print("\ny_train.info():")
    print("-" * 20)
    if hasattr(y_train, "info"):  # Series or DataFrame
        y_train.info()
    else:
        print(f"Type: {type(y_train)} — Cannot call .info()")

    # Sample data
    print("\nX_train head():")
    print(X_train.head())

    print("\ny_train head():")
    print(y_train.head())

    # Transformers used
    print("\nTransformers used:")
    for name, transformer in transformers.items():
        if isinstance(transformer, str):
            print(f"  - {name}: {transformer}")
        else:
            print(f"  - {name}: {type(transformer).__name__}")

    print("-" * 60)
