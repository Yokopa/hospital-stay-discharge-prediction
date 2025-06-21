from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss
)
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
    model_name, 
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
    # Resolve class string to actual class object
    clf_name = model_cfg[model_name]["classifier"]["class"]
    clf_class = config.CLASSIFIER_CLASSES[clf_name]
    clf_params = model_cfg[model_name]["classifier"].get("params", {}).copy()

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    # --- Handle imbalance for XGBoost (no SMOTE) ---
    if getattr(clf_class, "__name__", "") == "XGBClassifier":
        clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train)

    # --- Train classifier using helper ---
    classifier, y_pred, y_pred_proba = utils.train_classifier(
        clf_class, clf_params, X_train, y_train, X_test, y_test, categorical_features
    )

    # --- Metrics ---
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    roc_auc, logloss = None, None
    try:
        if hasattr(classifier, "predict_proba"):
            y_proba = classifier.predict_proba(X_test)
        else:
            y_proba = None

        if y_proba is not None and y_proba.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            logloss = log_loss(y_test, y_proba)
        elif y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            logloss = log_loss(y_test, y_proba)
    except Exception:
        pass

    return {
        "classifier": classifier,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "log_loss": logloss
    }