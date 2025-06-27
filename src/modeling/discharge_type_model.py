from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss
)
import numpy as np
import sys
import os
# Add parent directory to sys.path to find config.py one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils
import config


def train_discharge_pipeline(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    model_cfg
):
    """
    Train a classification pipeline for discharge type prediction.

    Args:
        X_train, X_test: pd.DataFrame or np.ndarray
            Train/test features.
        y_train, y_test: pd.Series or np.ndarray
            Train/test discharge type labels (binary or multi-class).
        model_name: str
            Model key (e.g., 'xgboost', 'catboost').
        model_cfg: dict
            Configuration for model class and parameters.

    Returns:
        dict: Contains trained model and classification metrics.
    """
    clf_name = model_cfg["classifier"]["class"]
    clf_class = config.CLASSIFIER_CLASSES[clf_name]
    clf_params = model_cfg["classifier"].get("params", {}).copy()

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    # Determine if sample weighting is enabled (default False)
    use_sample_weight = clf_params.pop("use_sample_weight", False)

    sample_weights = None

    if use_sample_weight:
        sample_weights = utils.compute_sample_weights(y_train)
            # Remove class_weight-related params to avoid conflicts
        clf_params.pop("class_weight", None)
        clf_params.pop("is_unbalance", None)
        clf_params.pop("scale_pos_weight", None)
        clf_params.pop("class_weights", None)  # just in case for CatBoost
    else:
        sample_weights = None

    # Train model using helper passing sample_weights if any
    classifier, y_pred, y_pred_proba = utils.train_classifier(
        clf_class, clf_params, X_train, y_train, X_test, y_test, categorical_features, sample_weights
    )

    # Compute weighted metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    # Compute macro metrics
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    # Other metrics
    cm = confusion_matrix(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    roc_auc, logloss = None, None
    try:
        if y_pred_proba is not None:
            if y_pred_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                logloss = log_loss(y_test, y_pred_proba)
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                logloss = log_loss(y_test, y_pred_proba)
    except Exception:
        pass

    return {
        "classifier": classifier,
        "f1_score": f1,
        "f1_macro": f1_macro,
        "precision": precision,
        "precision_macro": precision_macro,
        "recall": recall,
        "recall_macro": recall_macro,
        "confusion_matrix": cm,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "num_classes": len(np.unique(y_train))
    }

# def train_discharge_pipeline( # OLD VERSION, REMOVE IF THE NEW ONE WORKS
#     X_train, 
#     X_test, 
#     y_train, 
#     y_test, 
#     model_name, 
#     model_cfg
# ):
#     """
#     Train a classification pipeline for discharge type prediction.

#     Args:
#         X_train, X_test: pd.DataFrame or np.ndarray
#             Train/test features.
#         y_train, y_test: pd.Series or np.ndarray
#             Train/test discharge type labels (binary or multi-class).
#         model_name: str
#             Model key (e.g., 'xgboost', 'catboost').
#         model_cfg: dict
#             Configuration for model class and parameters.

#     Returns:
#         dict: Contains trained model and classification metrics.
#     """
#     # Resolve class string to actual class object
#     clf_name = model_cfg["classifier"]["class"]
#     clf_class = config.CLASSIFIER_CLASSES[clf_name]
#     clf_params = model_cfg["classifier"].get("params", {}).copy()

#     categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

# # -----------------------class imbalance handling-----------

#     # # --- Handle imbalance for XGBoost (no SMOTE) ---
#     # if getattr(clf_class, "__name__", "") == "XGBClassifier":
#     #     clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train)

#     # if getattr(clf_class, "__name__", "") == "LGBMClassifier":
#     #     clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train)

#     # elif clf_class.__name__ == "CatBoostClassifier":
#     #     clf_params["class_weights"] = utils.compute_class_weights(y_train)
    
#     # --- Train classifier using helper ---
#     classifier, y_pred, y_pred_proba = utils.train_classifier(
#         clf_class, clf_params, X_train, y_train, X_test, y_test, categorical_features
#     )

#     # --- Metrics ---
#     precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
#     cm = confusion_matrix(y_test, y_pred)
#     balanced_acc = balanced_accuracy_score(y_test, y_pred)

#     roc_auc, logloss = None, None
#     try:
#         if hasattr(classifier, "predict_proba"):
#             y_proba = classifier.predict_proba(X_test)
#         else:
#             y_proba = None

#         if y_proba is not None and y_proba.shape[1] == 2:
#             roc_auc = roc_auc_score(y_test, y_proba[:, 1])
#             logloss = log_loss(y_test, y_proba)
#         elif y_proba is not None:
#             roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
#             logloss = log_loss(y_test, y_proba)
#     except Exception:
#         pass

#     return {
#         "classifier": classifier,
#         "f1_score": f1,
#         "precision": precision,
#         "recall": recall,
#         "confusion_matrix": cm,
#         "balanced_accuracy": balanced_acc,
#         "roc_auc": roc_auc,
#         "log_loss": logloss
#     }
