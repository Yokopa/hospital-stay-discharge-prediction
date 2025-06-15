from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import config

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    balanced_accuracy_score, roc_auc_score, log_loss, classification_report
)
from sklearn.preprocessing import label_binarize

def train_discharge_pipeline(X_train, X_test, y_train, y_test, apply_smote=False, model_cfg=None, random_state=42):
    if model_cfg is None:
        model_cfg = config.DEFAULT_MODEL_CONFIG["random_forest"]  # default model

    # Optionally apply SMOTE on training set
    if apply_smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    clf_class = model_cfg.get("classifier", {}).get("class")
    clf_params = model_cfg.get("classifier", {}).get("params", {})
    model = clf_class(**clf_params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # Prepare for ROC AUC and log loss if probabilities available
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
        classes = model.classes_
        y_test_bin = label_binarize(y_test, classes=classes)
        roc_auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
        ll = log_loss(y_test, y_pred_proba)
    else:
        roc_auc = None
        ll = None

    return {
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "classification_report": report,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": cm,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "log_loss": ll,
    }
