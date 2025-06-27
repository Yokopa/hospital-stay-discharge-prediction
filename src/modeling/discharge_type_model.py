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

import logging
log = logging.getLogger(__name__)

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
        model_cfg: dict
            Configuration for model class and parameters.

    Returns:
        dict: Contains trained model and classification metrics.
    """
    log.info("Starting discharge type classification pipeline...")
    # Automatically handle class, params, and sample weights (incl. XGBoost-specific logic)
    clf_class, clf_params, sample_weights = utils.prepare_classifier_and_weights(
        model_cfg, y_train, y_train
    )

    # Detect categorical features
    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    # Train model
    classifier, y_pred, y_pred_proba = utils.train_classifier(
        clf_class, clf_params, 
        X_train, y_train, 
        X_test, y_test, 
        categorical_features, 
        sample_weight_train=sample_weights
    )
    log.info("Discharge type classification pipeline completed.")
    
    # Compute classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

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