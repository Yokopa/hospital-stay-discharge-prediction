"""
utils.py - Utility Functions for Data Handling, Preprocessing, and Modeling

This module includes reusable functions for:
- Loading/saving data and configs
- Preprocessing (e.g., negative value handling, statistics)
- Training models (classification, regression)
- Logging and evaluation utilities
"""

# === Imports ===
import pandas as pd
import numpy as np
from io import StringIO
from collections import Counter
import yaml
import logging
from joblib import dump
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import config

log = logging.getLogger(__name__)

# -----------------------------------#
# Data I/O Utilities
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

def configure_logging(verbose=True):
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# -----------------------------------#
# Preprocessing Utilities
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
    summary_stats = summary_stats.sort_values(by='missing_percentage', ascending=True)

    if verbose:
        log.info(f"Summary statistics shape: {summary_stats.shape}")
        log.info("Top columns by missingness:\n" + str(summary_stats.sort_values('missing_percentage', ascending=False).head(5)))

    # Save if path is provided
    if save_path:
        summary_stats.to_csv(save_path)

    return summary_stats

# -----------------------------------#
# Modeling Utilities
# -----------------------------------#

def compute_scale_pos_weight(y):
    """Compute scale_pos_weight = (neg / pos) for XGBoost."""
    counter = Counter(y)
    neg = counter.get(0, 0)
    pos = counter.get(1, 0)
    return neg / pos if pos > 0 else 1.0  # avoid division by zero

def compute_sample_weights(y):
    return compute_sample_weight(class_weight="balanced", y=y)

def train_classifier(clf_class, clf_params, X_train, y_train, X_test, y_test, cat_cols, sample_weight_train=None):
    """
    Train a classifier (CatBoost, LightGBM, or sklearn-compatible) and return predictions.

    Args:
        clf_class : class
            The classifier class (e.g., LGBMClassifier, CatBoostClassifier).
        clf_params : dict
            Parameters for the classifier.
        X_train, X_test : pd.DataFrame
            Feature sets.
        y_train, y_test : pd.Series or np.ndarray
            Targets.
        cat_cols : list
            Names of categorical columns.
        sample_weight_train : array-like, optional
            Sample weights for training data.

    Returns:
        classifier : fitted classifier
        y_pred : predicted classes
        y_pred_proba : predicted probabilities (if available, else None)
    """
    clf_name = clf_class.__name__
    log.info(f"Training classifier: {clf_name}...")

    if clf_name == "CatBoostClassifier":
        from catboost import Pool
        cat_indices = [X_train.columns.get_loc(col) for col in cat_cols]
        train_pool = Pool(X_train, y_train, cat_features=cat_indices, weight=sample_weight_train)
        test_pool = Pool(X_test, y_test, cat_features=cat_indices)
        clf = clf_class(**clf_params)
        clf.fit(train_pool, eval_set=test_pool, verbose=0)
        log.info("CatBoost training completed.")
        y_pred = clf.predict(test_pool)
        y_pred_proba = clf.predict_proba(test_pool)
        return clf, y_pred, y_pred_proba

    elif clf_name == "LGBMClassifier":
        clf = clf_class(**clf_params)
        fit_kwargs = {"categorical_feature": cat_cols}
        if sample_weight_train is not None:
            fit_kwargs["sample_weight"] = sample_weight_train
        clf.fit(X_train, y_train, **fit_kwargs)
        log.info("LightGBM training completed.")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        return clf, y_pred, y_pred_proba

    elif clf_name == "XGBClassifier":
        clf = clf_class(**clf_params)
        fit_kwargs = {}
        if sample_weight_train is not None and "scale_pos_weight" not in clf_params:
            fit_kwargs["sample_weight"] = sample_weight_train
        clf.fit(X_train, y_train, **fit_kwargs)
        log.info("XGBoost training completed.")
        y_pred = clf.predict(X_test)
        try:
            y_pred_proba = clf.predict_proba(X_test)
        except AttributeError:
            y_pred_proba = None
        return clf, y_pred, y_pred_proba

    else:
        clf = clf_class(**clf_params)
        fit_kwargs = {}
        if sample_weight_train is not None:
            fit_kwargs["sample_weight"] = sample_weight_train
        clf.fit(X_train, y_train, **fit_kwargs)
        log.info(f"{clf_name} training completed.")
        y_pred = clf.predict(X_test)
        try:
            y_pred_proba = clf.predict_proba(X_test)
        except AttributeError:
            y_pred_proba = None
        return clf, y_pred, y_pred_proba
    
def train_regressor(reg_class, reg_params, X_train, y_train, X_test, categorical_features):
    """
    Train a regressor (CatBoost, LightGBM, or sklearn-compatible) and return predictions.

    Args:
        reg_class : class
            The regressor class (e.g., LGBMRegressor, CatBoostRegressor).
        reg_params : dict
            Parameters for the regressor.
        X_train, X_test : pd.DataFrame
            Feature sets.
        y_train : pd.Series
            Target values.
        categorical_features : list
            Names of categorical columns.

    Returns:
        regressor : fitted model
        y_pred : predictions on X_test
    """
    log.info(f"Training regressor: {reg_class.__name__}")

    if reg_class.__name__ == "CatBoostRegressor":
        from catboost import Pool
        cat_indices = [X_train.columns.get_loc(col) for col in categorical_features]
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        test_pool = Pool(X_test, cat_features=cat_indices)
        regressor = reg_class(**reg_params)
        regressor.fit(train_pool)
        y_pred = regressor.predict(test_pool)

    elif reg_class.__name__ == "LGBMRegressor":
        regressor = reg_class(**reg_params)
        regressor.fit(X_train, y_train, categorical_feature=categorical_features)
        y_pred = regressor.predict(X_test)

    else:
        # fallback for scikit-learn-like regressors
        regressor = reg_class(**reg_params)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

    log.info(f"{reg_class.__name__} training completed.")
    return regressor, y_pred

def save_results(result, results_dir, model_prefix, now):
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([result]) if isinstance(result, dict) else pd.DataFrame(result)
    out_path = results_dir / f"{model_prefix}_results_{now}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved results to {out_path}")

    if "confusion_matrix" in result:
        cm_path = results_dir / f"{model_prefix}_confusion_matrix.csv"
        pd.DataFrame(result["confusion_matrix"]).to_csv(cm_path, index=False)
        log.info(f"Saved confusion matrix to {cm_path}")

def save_models(result, models_dir, model_prefix):
    models_dir.mkdir(parents=True, exist_ok=True)
    if "classifier" in result:
        clf_path = models_dir / f"{model_prefix}_classifier.joblib"
        dump(result["classifier"], clf_path)
        log.info(f"Saved classifier to {clf_path}")
    if "regressor" in result:
        reg_path = models_dir / f"{model_prefix}_regressor.joblib"
        dump(result["regressor"], reg_path)
        log.info(f"Saved regressor to {reg_path}")

def compute_per_class_metrics(y_preds):
    rmse_per_class = {}
    mae_per_class = {}
    r2_per_class = {}
    all_true = []
    all_pred = []

    for pred in y_preds:
        cls = pred.cls
        y_true = pred.y_true
        y_pred = pred.y_pred

        if len(y_true) > 0:
            rmse = root_mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            rmse_per_class[cls] = rmse
            mae_per_class[cls] = mae
            r2_per_class[cls] = r2

            all_true.append(y_true)
            all_pred.append(y_pred)

    if all_true and all_pred:
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        overall_metrics = {
            "rmse": root_mean_squared_error(all_true, all_pred),
            "mae": mean_absolute_error(all_true, all_pred),
            "r2": r2_score(all_true, all_pred)
        }
    else:
        overall_metrics = {"rmse": None, "mae": None, "r2": None}

    return rmse_per_class, mae_per_class, r2_per_class, overall_metrics


def log_df_info(df, name="DataFrame"):
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    log.info(f"{name} info:\n{info_str}")

def prepare_classifier_and_weights(model_cfg, y_train, y_train_cls=None):
    """
    Extract classifier class, params, and compute sample weights if needed.
    Handles special case for XGBClassifier.

    Returns:
        clf_class: the model class (e.g., XGBClassifier)
        clf_params: dict of cleaned params
        sample_weights: np.array or None
    """
    clf_name = model_cfg["classifier"]["class"]
    clf_params = model_cfg["classifier"].get("params", {}).copy()
    clf_params["random_state"] = config.RANDOM_SEED
    clf_class = config.CLASSIFIER_CLASSES[clf_name]

    use_sample_weight = model_cfg["classifier"].get("use_sample_weight", False)
    sample_weights = None

    y_target = y_train_cls if y_train_cls is not None else y_train
    num_classes = len(np.unique(y_target))

    if clf_name == "XGBClassifier" and use_sample_weight:
        if num_classes == 2:
            log.info("Applying XGBoost class balancing with scale_pos_weight.")
            clf_params["scale_pos_weight"] = compute_scale_pos_weight(y_target)
        else:
            log.info("Applying XGBoost sample weighting for multi-class classification.")
            sample_weights = compute_sample_weights(y_target)
    elif use_sample_weight:
        log.info(f"Applying sample weighting for classifier: {clf_name}")
        sample_weights = compute_sample_weights(y_target)

    return clf_class, clf_params, sample_weights
