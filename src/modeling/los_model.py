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

def train_los_regression_baseline(X_train, X_test, y_train, y_test, model_cfg):
    reg_name = model_cfg["regressor"]["class"]  # e.g. "LGBMRegressor"
    reg_params = model_cfg["regressor"].get("params", {}).copy()
    reg_params["random_state"] = config.RANDOM_SEED

    if reg_name not in config.REGRESSOR_CLASSES:
        raise ValueError(f"Unknown regressor class: {reg_name}")

    reg_class = config.REGRESSOR_CLASSES[reg_name]
    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    regressor, y_pred_reg = utils.train_regressor(
        reg_class, reg_params, X_train, y_train, X_test, categorical_features
    )
    rmse = root_mean_squared_error(y_test, y_pred_reg)
    mae = mean_absolute_error(y_test, y_pred_reg)
    r2 = r2_score(y_test, y_pred_reg)

    return {
        "regressor": regressor,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def train_los_multiclass_baseline(X_train, X_test, y_train, y_test, bins, model_cfg):
    y_train_cls = pd.cut(y_train, bins=bins, labels=False, right=False)
    y_test_cls = pd.cut(y_test, bins=bins, labels=False, right=False)

    clf_name = model_cfg["classifier"]["class"]  # e.g. "LGBMClassifier"
    clf_params = model_cfg["classifier"].get("params", {}).copy()
    clf_params["random_state"] = config.RANDOM_SEED

    if clf_name not in config.CLASSIFIER_CLASSES:
        raise ValueError(f"Unknown classifier class: {clf_name}")

    clf_class = config.CLASSIFIER_CLASSES[clf_name]

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    classifier, y_pred_cls, _ = utils.train_classifier(
        clf_class, clf_params, X_train, y_train_cls, X_test, y_test_cls, categorical_features
    )

    f1 = f1_score(y_test_cls, y_pred_cls, average='weighted')
    precision = precision_score(y_test_cls, y_pred_cls, average='weighted')
    recall = recall_score(y_test_cls, y_pred_cls, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test_cls, y_pred_cls)
    cm = confusion_matrix(y_test_cls, y_pred_cls)

    return {
        "classifier": classifier,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "num_classes": len(np.unique(y_train_cls))
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

    from config import (
        RANDOM_SEED, CLASSIFIER_CLASSES, REGRESSOR_CLASSES
    )
    import utils

    # Step 1: Classify using thresholds (binary or multiclass)
    thresholds = sorted(thresholds)
    y_train_cls = np.digitize(y_train, thresholds)
    y_test_cls = np.digitize(y_test, thresholds)

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    # --- CLASSIFIER ---
    clf_name = model_cfg["classifier"]["class"]
    clf_params = model_cfg["classifier"]["params"].copy()
    clf_params["random_state"] = RANDOM_SEED
    clf_class = CLASSIFIER_CLASSES[clf_name]

    if getattr(clf_class, "__name__", "") == "XGBClassifier":
        clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train_cls)

    classifier, y_pred_cls, _ = utils.train_classifier(
        clf_class, clf_params, X_train, y_train_cls, X_test, y_test_cls, categorical_features
    )

    precision, recall, f1, _ = precision_recall_fscore_support(y_test_cls, y_pred_cls, average='weighted')
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
    except Exception as e:
        warnings.warn(f"ROC AUC or Log Loss computation failed: {e}")

    # --- REGRESSORS ---
    reg_name = model_cfg["regressor"]["class"]
    reg_params = model_cfg["regressor"]["params"].copy()
    reg_params["random_state"] = RANDOM_SEED
    reg_class = REGRESSOR_CLASSES[reg_name]

    regressors = {}
    y_preds = []

    for cls in np.unique(y_train_cls):
        cls_mask_train = y_train_cls == cls
        cls_mask_test = y_pred_cls == cls

        # Skip if no samples for this class in train or test
        if cls_mask_train.sum() == 0 or cls_mask_test.sum() == 0:
            warnings.warn(f"Class {cls}: No samples in train or predicted test set. Skipping regressor for this class.")
            continue

        regressor, y_pred_reg = utils.train_regressor(
            reg_class, reg_params, 
            X_train[cls_mask_train], y_train[cls_mask_train], 
            X_test[cls_mask_test], 
            categorical_features
        )
        regressors[int(cls)] = regressor

        pred_df = Bunch(
            cls=int(cls),
            y_true=np.array(y_test[cls_mask_test]),
            y_pred=np.array(y_pred_reg)
        )
        y_preds.append(pred_df)

    # --- METRICS ---
    rmse_per_class, mae_per_class, r2_per_class, overall_metrics = utils.compute_per_class_metrics(y_preds)


    return {
        "classifier": classifier,
        "regressors": regressors,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "rmse": overall_metrics["rmse"],
        "mae": overall_metrics["mae"],
        "r2": overall_metrics["r2"],
        "rmse_per_class": rmse_per_class,
        "mae_per_class": mae_per_class,
        "r2_per_class": r2_per_class,
        "thresholds": thresholds
}


# def train_los_two_step_pipeline( #OLD VERSION
#     X_train, X_test, y_train, y_test, threshold, model_name, model_cfg
# ):
#     y_train_cls = (y_train > threshold).astype(int)
#     y_test_cls = (y_test > threshold).astype(int)

#     X_train_cls, X_test_cls = X_train.copy(), X_test.copy()

#     categorical_features = X_train_cls.select_dtypes(include=["category", "object"]).columns.tolist()

#     # --- Use model_name to access correct config sub-dict ---
#     clf_name = model_cfg["classifier"]["class"]
#     clf_params = model_cfg["classifier"]["params"].copy()
#     clf_params["random_state"] = config.RANDOM_SEED

#     clf_class = config.CLASSIFIER_CLASSES[clf_name]

#     if getattr(clf_class, "__name__", "") == "XGBClassifier":
#         clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train_cls)

#     classifier, y_pred_cls, _ = utils.train_classifier(
#         clf_class, clf_params, X_train, y_train_cls, X_test, y_test_cls, categorical_features
#     )

#     precision, recall, f1, _ = precision_recall_fscore_support(y_test_cls, y_pred_cls, average='weighted')
#     cm = confusion_matrix(y_test_cls, y_pred_cls)
#     balanced_acc = balanced_accuracy_score(y_test_cls, y_pred_cls)

#     roc_auc = None
#     logloss = None
#     try:
#         if hasattr(classifier, "predict_proba"):
#             y_pred_proba = classifier.predict_proba(X_test_cls)
#         else:
#             y_pred_proba = None

#         if y_pred_proba is not None:
#             if y_pred_proba.shape[1] == 2:
#                 roc_auc = roc_auc_score(y_test_cls, y_pred_proba[:, 1])
#             else:
#                 roc_auc = roc_auc_score(y_test_cls, y_pred_proba, multi_class='ovr')
#             logloss = log_loss(y_test_cls, y_pred_proba)
#     except Exception as e:
#         print(f"[Warning] ROC AUC or Log Loss computation failed: {e}")

#     reg_name = model_cfg["regressor"]["class"]
#     reg_params = model_cfg["regressor"]["params"].copy()
#     reg_params["random_state"] = config.RANDOM_SEED

#     reg_class = config.REGRESSOR_CLASSES[reg_name]
#     regressor = reg_class(**reg_params)
#     regressor, y_pred_reg = utils.train_regressor(
#     reg_class, reg_params, X_train, y_train, X_test, categorical_features
#     )
#     # regressor.fit(X_train, y_train)
#     # y_pred_reg = regressor.predict(X_test)

#     rmse = root_mean_squared_error(y_test, y_pred_reg)
#     mae = mean_absolute_error(y_test, y_pred_reg)
#     r2 = r2_score(y_test, y_pred_reg)

#     rmse_short, rmse_long = None, None
#     try:
#         short_mask = y_test_cls == 0
#         long_mask = y_test_cls == 1
#         if short_mask.any():
#             rmse_short = root_mean_squared_error(y_test[short_mask], y_pred_reg[short_mask])
#         if long_mask.any():
#             rmse_long = root_mean_squared_error(y_test[long_mask], y_pred_reg[long_mask])
#     except Exception as e:
#         print(f"[Warning] RMSE split computation failed: {e}")

#     return {
#         "classifier": classifier,
#         "regressor": regressor,
#         "f1_score": f1,
#         "precision": precision,
#         "recall": recall,
#         "confusion_matrix": cm,
#         "balanced_accuracy": balanced_acc,
#         "roc_auc": roc_auc,
#         "log_loss": logloss,
#         "rmse": rmse,
#         "mae": mae,
#         "r2": r2,
#         "rmse_short": rmse_short,
#         "rmse_long": rmse_long,
#         "threshold": threshold
#     }

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

# ONLY BINARY
# def compare_los_thresholds(X_train, X_test, y_train, y_test, thresholds=[7, 14, 30], model_cfg=None, model_name=None, dataset_name=None):
#     results = []
#     for th in thresholds:
#         result = train_los_two_step_pipeline(
#             X_train, X_test, y_train, y_test,
#             threshold=th,
#             model_name=model_name,
#             model_cfg=model_cfg
#         )
#         results.append({
#             "Dataset": dataset_name,
#             "Threshold": th,
#             "F1 Score": result["f1_score"],
#             "Precision": result["precision"],
#             "Recall": result["recall"],
#             "Balanced Accuracy": result["balanced_accuracy"],
#             "ROC AUC": result["roc_auc"],
#             "Log Loss": result["log_loss"],
#             "RMSE (Overall)": result["rmse"],
#             "MAE (Overall)": result["mae"],
#             "R2 Score": result["r2"],
#             "RMSE (Short Stay)": result.get("rmse_short"),
#             "RMSE (Long Stay)": result.get("rmse_long"),
#         })
#     return pd.DataFrame(results)