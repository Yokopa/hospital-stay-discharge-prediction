"""
utils.py - Utility Functions for Data Handling, Preprocessing, and Modeling

This module includes reusable functions for:
- Loading/saving data and configs
- Preprocessing (e.g., negative value handling, statistics)
- Training models (classification, regression)
- Logging and evaluation utilities
"""

# === Imports ===
import logging
from collections import Counter
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import shap
import matplotlib.pyplot as plt

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
    # Only select numeric columns
    numeric_df = df.select_dtypes(include='number')
    if start_col is None:
        start_col = 0
    if end_col is None:
        end_col = len(numeric_df.columns)
    cols = numeric_df.columns[start_col:end_col]
    negative_columns = cols[(numeric_df[cols] < 0).any()]
    num_negative_entries = (numeric_df.loc[:, negative_columns] < 0).sum().sum()
    log.info(f"Total negative entries in numeric columns: {num_negative_entries}")
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
    for col in columns:
        df[col] = df[col].astype(float).map(lambda x: np.nan if x < 0 else x)
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

def missing_value_summary(df, name="DataFrame", top_n=10):
    """
    Print and log a summary of missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        name (str): Name for logging.
        top_n (int): Number of top columns by missing values to display.
    """
    total_missing = df.isnull().sum().sum()
    missing_per_col = df.isnull().sum()
    missing_cols = missing_per_col[missing_per_col > 0].sort_values(ascending=False)
    log.info(f"{name}: Total missing values: {total_missing}")
    if not missing_cols.empty:
        log.info(f"{name}: Columns with missing values (top {top_n}):\n{missing_cols.head(top_n)}")
    else:
        log.info(f"{name}: No missing values.")

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

def get_filename_suffix(args):
    suffix_parts = []

    if args.target == "discharge_type":
        suffix_parts.append(f"{config.DISCHARGE_CATEGORIES_NUMBER}cat")

    if args.target == "los":
        method = config.LOS_TRANSFORMATION.get("method", "none")

        if method == "cap":
            cap_value = config.LOS_TRANSFORMATION.get("cap_value", "")
            suffix_parts.append(f"cap{cap_value}")  # "cap99", no need to also add "cap"
        elif method == "winsorize":
            winsor_limits = config.LOS_TRANSFORMATION.get("winsor_limits", "")
            suffix_parts.append(f"wins{winsor_limits}")
        elif method == "log":
            suffix_parts.append("log")  # add only once
        elif method != "none":
            suffix_parts.append(method)  # fallback for other methods

        if args.mode:
            suffix_parts.append(args.mode)

        if args.mode in ["multiclass", "two_step"] and args.thresholds:
            thresh_str = "_".join(map(str, args.thresholds))
            suffix_parts.append(f"thresh_{thresh_str}")

    return "_".join(suffix_parts)

def save_model_outputs( # Old version saved on the cluster
    result: dict,
    model_type: str,
    model_name: str,
    param_set: str,
    dataset_name: str,
    filename_suffix: str,
    results_dir: Path,
    models_dir: Path,
    save_shap: bool = True,
    save_preds: bool = True,
    save_model: bool = False
):
    """
    Save model outputs including metrics, trained model(s), predictions, confusion matrix, and SHAP values/plots.

    Supports discharge type classification, LOS classification, LOS regression, and two-step LOS models.
    """

    # Ensure directories exist
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Build filename prefix
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{model_type}_{model_name}_{param_set}_{dataset_name}"
    if filename_suffix:
        prefix += f"_{filename_suffix}"
    prefix += f"_{timestamp}"

    # --- Save metrics ---
    metrics_keys = [k for k in result.keys()
                if k not in ("classifier", "regressor", "regressors", "X_test", "confusion_matrix")
                and not k.startswith(("y_", "shap"))]
    metrics = {k: result[k] for k in metrics_keys}
    df_metrics = pd.DataFrame([metrics])
    df_metrics["dataset_name"] = dataset_name
    df_metrics.to_csv(results_dir / f"{prefix}_metrics.csv", index=False)

    # --- Save model(s) ---
    if save_model:
        if "classifier" in result:
            dump(result["classifier"], models_dir / f"{prefix}_classifier.joblib")
        if "regressor" in result:
            dump(result["regressor"], models_dir / f"{prefix}_regressor.joblib")
        if "regressors" in result:
            for cls, reg in result["regressors"].items():
                dump(reg, models_dir / f"{prefix}_regressor_class{cls}.joblib")

    # --- Save predictions ---
    if save_preds:
        for name in ["y_test", "y_pred", "y_pred_proba"]:
            if name in result and result[name] is not None:
                pd.DataFrame(result[name]).to_csv(results_dir / f"{prefix}_{name}.csv", index=False)

    # --- Save confusion matrix ---
    if "confusion_matrix" in result:
        pd.DataFrame(result["confusion_matrix"]).to_csv(results_dir / f"{prefix}_confusion_matrix.csv", index=False)

    # --- SHAP for classifier ---
    if save_shap and "classifier" in result and "X_test" in result and hasattr(result["classifier"], "predict_proba"):
        try:
            explainer = shap.Explainer(result["classifier"])
            shap_values = explainer(result["X_test"])
            dump(shap_values, results_dir / f"{prefix}_shap_classifier.joblib")
            shap.summary_plot(shap_values, result["X_test"], show=False)
            plt.tight_layout()
            plt.savefig(results_dir / f"{prefix}_shap_classifier_summary.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not compute SHAP for classifier: {e}")

    # --- SHAP for regression ---
    if save_shap and "regressor" in result and "X_test" in result:
        try:
            explainer = shap.Explainer(result["regressor"])
            shap_values = explainer(result["X_test"])
            dump(shap_values, results_dir / f"{prefix}_shap_regressor.joblib")
            shap.summary_plot(shap_values, result["X_test"], show=False)
            plt.tight_layout()
            plt.savefig(results_dir / f"{prefix}_shap_regressor_summary.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not compute SHAP for regressor: {e}")

    # --- SHAP for each per-class regressor (two-step) ---
    if save_shap and "regressors" in result and "X_test" in result and "y_pred" in result:
        try:
            y_pred_cls = result["y_pred"]
            X_test = result["X_test"]
            for cls, reg in result["regressors"].items():
                cls_mask = (y_pred_cls == cls)
                if cls_mask.sum() == 0:
                    continue
                explainer = shap.Explainer(reg)
                shap_values = explainer(X_test[cls_mask])
                dump(shap_values, results_dir / f"{prefix}_shap_regressor_class{cls}.joblib")
                shap.summary_plot(shap_values, X_test[cls_mask], show=False)
                plt.tight_layout()
                plt.savefig(results_dir / f"{prefix}_shap_regressor_class{cls}_summary.png")
                plt.close()
        except Exception as e:
            print(f"[Warning] Could not compute SHAP for per-class regressors: {e}")
