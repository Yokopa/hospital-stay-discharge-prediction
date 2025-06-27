from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    balanced_accuracy_score, roc_auc_score, log_loss,
    mean_absolute_error, r2_score, root_mean_squared_error,
    f1_score, precision_score, recall_score
)
from sklearn.utils import Bunch
import warnings
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import utils

import logging
log = logging.getLogger(__name__)

def train_los_regression_baseline(X_train, X_test, y_train, y_test, model_cfg):
    log.info("Starting LOS regression baseline training...")
    reg_name = model_cfg["regressor"]["class"]  # e.g. "LGBMRegressor"
    reg_params = model_cfg["regressor"].get("params", {}).copy()
    reg_params["random_state"] = config.RANDOM_SEED

    if reg_name not in config.REGRESSOR_CLASSES:
        raise ValueError(f"Unknown regressor class: {reg_name}")

    reg_class = config.REGRESSOR_CLASSES[reg_name]
    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    log.info(f"Training regressor: {reg_class.__name__}")
    regressor, y_pred_reg = utils.train_regressor(
        reg_class, reg_params, X_train, y_train, X_test, categorical_features
    )
    log.info("Regressor training completed.")

    # Inverse transform predictions if log transform was applied on target
    if config.LOS_TRANSFORMATION.get("method", "none") == "log":
        y_pred_reg = np.expm1(y_pred_reg)  # inverse of np.log1p
        y_test_inv = np.expm1(y_test)
    else:
        y_test_inv = y_test

    rmse = root_mean_squared_error(y_test_inv, y_pred_reg)
    mae = mean_absolute_error(y_test_inv, y_pred_reg)
    r2 = r2_score(y_test_inv, y_pred_reg)

    log.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {
        "regressor": regressor,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def log_los_stats_per_class(y_true, y_binned, name="Train"):
    log.info(f"LOS stats per class ({name}):")
    df = pd.DataFrame({"los": y_true, "class": y_binned})
    stats = df.groupby("class")["los"].describe()[["min", "25%", "50%", "75%", "max", "mean"]]
    for cls, row in stats.iterrows():
        log.info(f"  Class {cls}: {row.to_dict()}")

def train_los_multiclass_baseline(X_train, X_test, y_train, y_test, bins, model_cfg):
    log.info("Starting multiclass LOS baseline classification...")
    y_train_cls = pd.cut(y_train, bins=bins, labels=False, right=False)
    y_test_cls = pd.cut(y_test, bins=bins, labels=False, right=False)
    log.info("Class distribution after binning:")
    log.info(f"Train: {pd.Series(y_train_cls).value_counts().sort_index().to_dict()}")
    log.info(f"Test: {pd.Series(y_test_cls).value_counts().sort_index().to_dict()}")
    log_los_stats_per_class(y_train, y_train_cls, "Train")
    log_los_stats_per_class(y_test, y_test_cls, "Test")

    # Get classifier, params, and sample weights using utility
    clf_class, clf_params, sample_weights = utils.prepare_classifier_and_weights(
        model_cfg, y_train, y_train_cls
    )

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    # TRAIN using y_train_cls (classified labels)
    log.info(f"Training classifier: {clf_class.__name__}")
    classifier, y_pred_cls, y_pred_proba = utils.train_classifier(
        clf_class, clf_params,
        X_train, y_train_cls,
        X_test, y_test_cls,
        categorical_features,
        sample_weight_train=sample_weights
    )
    log.info("Multiclass classifier training completed.")

    # METRICS
    f1_weighted = f1_score(y_test_cls, y_pred_cls, average='weighted')
    precision_weighted = precision_score(y_test_cls, y_pred_cls, average='weighted')
    recall_weighted = recall_score(y_test_cls, y_pred_cls, average='weighted')

    f1_macro = f1_score(y_test_cls, y_pred_cls, average='macro')
    precision_macro = precision_score(y_test_cls, y_pred_cls, average='macro')
    recall_macro = recall_score(y_test_cls, y_pred_cls, average='macro')

    balanced_acc = balanced_accuracy_score(y_test_cls, y_pred_cls)
    cm = confusion_matrix(y_test_cls, y_pred_cls)

    return {
        "classifier": classifier,
        "f1_score": f1_weighted,
        "precision": precision_weighted,
        "recall": recall_weighted,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "num_classes": len(np.unique(y_train_cls)),
        "thresholds": bins
    }

def train_los_two_step_pipeline(
    X_train, X_test, y_train, y_test, thresholds, model_cfg
):
    """
    Two-step LOS prediction pipeline: 
    1. Classify LOS into bins (binary or multiclass, based on thresholds).
    2. For each predicted class, train a separate regressor and predict LOS.

    Args:
        X_train, X_test: pd.DataFrame
            Feature sets for training and testing.
        y_train, y_test: pd.Series or np.ndarray
            LOS values for training and testing.
        thresholds: list of float
            Thresholds to bin LOS for classification (e.g., [7] for binary, [7, 14, 30] for multiclass).
        model_name: str
            Model key (e.g., 'xgboost', 'catboost').
        model_cfg: dict
            Configuration for classifier and regressor.

    Returns:
        dict: Contains classifier, regressors, and all relevant metrics.
    """
    log.info("Starting two-step LOS prediction pipeline...")
    # Classify using thresholds (binary or multiclass)
    thresholds = sorted(thresholds)
    y_train_cls = np.digitize(y_train, thresholds)
    y_test_cls = np.digitize(y_test, thresholds)
    log.info("LOS classification class distribution (two-step):")
    log.info(f"Train: {pd.Series(y_train_cls).value_counts().sort_index().to_dict()}")
    log.info(f"Test: {pd.Series(y_test_cls).value_counts().sort_index().to_dict()}")
    log_los_stats_per_class(y_train, y_train_cls, "Train")
    log_los_stats_per_class(y_test, y_test_cls, "Test")

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

     # --- CLASSIFIER ---
    log.info("Preparing classifier and sample weights...")
    clf_class, clf_params, sample_weights = utils.prepare_classifier_and_weights(
        model_cfg, y_train, y_train_cls
    )

    # Now train
    classifier, y_pred_cls, _ = utils.train_classifier(
        clf_class, clf_params,
        X_train, y_train_cls,
        X_test, y_test_cls,
        categorical_features,
        sample_weight_train=sample_weights
    )

    precision, recall, f1, _ = precision_recall_fscore_support(y_test_cls, y_pred_cls, average='weighted')
    # add macro
    f1_macro = f1_score(y_test_cls, y_pred_cls, average='macro')
    precision_macro = precision_score(y_test_cls, y_pred_cls, average='macro')
    recall_macro = recall_score(y_test_cls, y_pred_cls, average='macro')
    cm = confusion_matrix(y_test_cls, y_pred_cls)
    balanced_acc = balanced_accuracy_score(y_test_cls, y_pred_cls)

    roc_auc = None
    logloss = None
    try:
        if hasattr(classifier, "predict_proba"):
            y_pred_proba = classifier.predict_proba(X_test)
            if y_pred_proba.shape[1] == len(np.unique(y_test_cls)):
                roc_auc = roc_auc_score(y_test_cls, y_pred_proba, multi_class='ovr')
                logloss = log_loss(y_test_cls, y_pred_proba)
                log.info(f"Log Loss computed: {logloss:.4f}")
    except Exception as e:
        warnings.warn(f"ROC AUC or Log Loss computation failed: {e}")

    # --- REGRESSORS ---
    log.info("🔧 Training per-class regressors...")
    reg_name = model_cfg["regressor"]["class"]
    reg_params = model_cfg["regressor"]["params"].copy()
    reg_params["random_state"] = config.RANDOM_SEED
    reg_class = config.REGRESSOR_CLASSES[reg_name]

    regressors = {}
    y_preds = []

    for cls in np.unique(y_train_cls):
        cls_mask_train = y_train_cls == cls
        cls_mask_test = y_pred_cls == cls

        if cls_mask_train.sum() == 0 or cls_mask_test.sum() == 0:
            warnings.warn(f"Class {cls}: No samples in train or predicted test set. Skipping regressor for this class.")
            continue

        log.info(f"Training regressor for class {cls}...")
        regressor, y_pred_reg = utils.train_regressor(
            reg_class, reg_params, 
            X_train[cls_mask_train], y_train[cls_mask_train], 
            X_test[cls_mask_test], 
            categorical_features
        )
        
        # Inverse transform predictions and true values if log transform applied
        if config.LOS_TRANSFORMATION.get("method", "none") == "log":
            y_pred_reg = np.expm1(y_pred_reg)
            y_true_cls = np.expm1(y_test[cls_mask_test])
        else:
            y_true_cls = y_test[cls_mask_test]

        regressors[int(cls)] = regressor

        pred_df = Bunch(
            cls=int(cls),
            y_true=np.array(y_true_cls),
            y_pred=np.array(y_pred_reg)
        )
        y_preds.append(pred_df)

    # --- METRICS ---
    log.info("Computing regression metrics...")
    rmse_per_class, mae_per_class, r2_per_class, overall_metrics = utils.compute_per_class_metrics(y_preds)

    log.info("Two-step LOS prediction pipeline completed.")

    return {
        "classifier": classifier,
        "regressors": regressors,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "rmse": overall_metrics["rmse"],
        "mae": overall_metrics["mae"],
        "r2": overall_metrics["r2"],
        "rmse_per_class": rmse_per_class,
        "mae_per_class": mae_per_class,
        "r2_per_class": r2_per_class,
        "thresholds": thresholds,
        "num_classes": len(np.unique(y_train_cls))
}

def compare_los_thresholds(
    X_train, X_test, y_train, y_test,
    threshold_sets=config.LOS_TARGET_THRESHOLDS,  # list of lists of thresholds
    model_cfg=None,
    model_name=None,
    dataset_name=None
):
    results = []
    for thresholds in threshold_sets:
        result = train_los_two_step_pipeline(
            X_train, X_test, y_train, y_test,
            thresholds=thresholds,
            model_cfg=model_cfg
        )
        row = {
            "Dataset": dataset_name,
            "Thresholds": thresholds,
            "F1 Score": result["f1_score"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "Balanced Accuracy": result["balanced_accuracy"],
            "ROC AUC": result["roc_auc"],
            "Log Loss": result["log_loss"],
            "RMSE (Overall)": result["rmse"],
            "MAE (Overall)": result["mae"],
            "R2 Score": result["r2"],
        }
        if "rmse_per_class" in result:
            for cls, rmse_val in result["rmse_per_class"].items():
                row[f"RMSE (Class {cls})"] = rmse_val
        results.append(row)
    return pd.DataFrame(results)
