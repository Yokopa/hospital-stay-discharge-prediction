from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    balanced_accuracy_score, roc_auc_score, log_loss,
    mean_absolute_error, r2_score, root_mean_squared_error,
    f1_score, precision_score, recall_score
)
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
    regressor = reg_class(**reg_params)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

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
    X_train, X_test, y_train, y_test, threshold, model_name, model_cfg
):
    y_train_cls = (y_train > threshold).astype(int)
    y_test_cls = (y_test > threshold).astype(int)

    X_train_cls, X_test_cls = X_train.copy(), X_test.copy()

    categorical_features = X_train_cls.select_dtypes(include=["category", "object"]).columns.tolist()

    # --- FIX: Use model_name to access correct config sub-dict ---
    clf_name = model_cfg[model_name]["classifier"]["class"]
    clf_params = model_cfg[model_name]["classifier"]["params"].copy()
    clf_params["random_state"] = config.RANDOM_SEED

    clf_class = config.CLASSIFIER_CLASSES[clf_name]

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
            y_pred_proba = classifier.predict_proba(X_test_cls)
        else:
            y_pred_proba = None

        if y_pred_proba is not None:
            if y_pred_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test_cls, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test_cls, y_pred_proba, multi_class='ovr')
            logloss = log_loss(y_test_cls, y_pred_proba)
    except Exception as e:
        print(f"[Warning] ROC AUC or Log Loss computation failed: {e}")

    reg_name = model_cfg[model_name]["regressor"]["class"]
    reg_params = model_cfg[model_name]["regressor"]["params"].copy()
    reg_params["random_state"] = config.RANDOM_SEED

    reg_class = config.REGRESSOR_CLASSES[reg_name]
    regressor = reg_class(**reg_params)
    regressor.fit(X_train, y_train)
    y_pred_reg = regressor.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred_reg)
    mae = mean_absolute_error(y_test, y_pred_reg)
    r2 = r2_score(y_test, y_pred_reg)

    rmse_short, rmse_long = None, None
    try:
        short_mask = y_test_cls == 0
        long_mask = y_test_cls == 1
        if short_mask.any():
            rmse_short = root_mean_squared_error(y_test[short_mask], y_pred_reg[short_mask])
        if long_mask.any():
            rmse_long = root_mean_squared_error(y_test[long_mask], y_pred_reg[long_mask])
    except Exception as e:
        print(f"[Warning] RMSE split computation failed: {e}")

    return {
        "classifier": classifier,
        "regressor": regressor,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmse_short": rmse_short,
        "rmse_long": rmse_long,
        "threshold": threshold
    }