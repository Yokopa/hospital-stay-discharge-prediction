from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    balanced_accuracy_score, roc_auc_score, log_loss, mean_absolute_error, r2_score,
    root_mean_squared_error
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import config
from imblearn.over_sampling import SMOTE

def train_los_pipeline(X_train, X_test, y_train, y_test, threshold=7, apply_smote=False, model_cfg=None, random_state=42):
    if model_cfg is None:
        model_cfg = config.DEFAULT_MODEL_CONFIG

    y_train_cls = (y_train > threshold).astype(int)
    y_test_cls = (y_test > threshold).astype(int)

    X_train_cls = X_train.copy()
    X_test_cls = X_test.copy()

    if apply_smote:
        sm = SMOTE(random_state=random_state)
        X_train_cls, y_train_cls = sm.fit_resample(X_train_cls, y_train_cls)

    clf_class = model_cfg.get("classifier", {}).get("class", RandomForestClassifier)
    clf_params = model_cfg.get("classifier", {}).get("params", {})
    classifier = clf_class(**clf_params)
    classifier.fit(X_train_cls, y_train_cls)
    y_pred_cls = classifier.predict(X_test_cls)

    # Extra classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_cls, y_pred_cls, average='weighted')
    cm = confusion_matrix(y_test_cls, y_pred_cls)
    balanced_acc = balanced_accuracy_score(y_test_cls, y_pred_cls)
    roc_auc = None
    logloss = None
    if hasattr(classifier, "predict_proba"):
        y_pred_proba = classifier.predict_proba(X_test_cls)
        roc_auc = roc_auc_score(y_test_cls, y_pred_proba[:,1])
        logloss = log_loss(y_test_cls, y_pred_proba)

    # Regression step
    reg_class = model_cfg.get("regressor", {}).get("class", RandomForestRegressor)
    reg_params = model_cfg.get("regressor", {}).get("params", {})
    regressor = reg_class(**reg_params)
    regressor.fit(X_train, y_train)
    y_pred_reg = regressor.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred_reg, squared=False)
    mae = mean_absolute_error(y_test, y_pred_reg)
    r2 = r2_score(y_test, y_pred_reg)

    # RMSE per group (short, long)
    rmse_short, rmse_long = None, None
    try:
        short_mask = y_test_cls == 0
        long_mask = y_test_cls == 1
        if short_mask.any():
            rmse_short = root_mean_squared_error(y_test[short_mask], y_pred_reg[short_mask], squared=False)
        if long_mask.any():
            rmse_long = root_mean_squared_error(y_test[long_mask], y_pred_reg[long_mask], squared=False)
    except Exception:
        pass

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


def compare_los_thresholds(X, y_los_days, thresholds=[7, 14, 30], apply_smote=True, model_cfg=None):
    """
    Train and compare LOS models across multiple thresholds.
    Returns a DataFrame with multiple performance metrics.
    """
    results = []
    for th in thresholds:
        result = train_los_pipeline(X, y_los_days, threshold=th, apply_smote=apply_smote, model_cfg=model_cfg)
        results.append({
            "Threshold": th,
            "F1 Score": result["f1_score"],
            "Precision": result["precision"],
            "Recall": result["recall"],
            "Balanced Accuracy": result["balanced_accuracy"],
            "ROC AUC": result["roc_auc"],
            "Log Loss": result["log_loss"],
            "RMSE (Overall)": result["rmse"],
            "MAE (Overall)": result["mae"],
            "R2 Score": result["r2"],
            "RMSE (Short Stay)": result.get("rmse_short"),
            "RMSE (Long Stay)": result.get("rmse_long"),
        })

    return pd.DataFrame(results)


def compare_los_results(results_df):
    """
    Print a summary of classification and regression metrics per threshold.
    """
    for _, row in results_df.iterrows():
        print(f"\n=== Threshold: {row['Threshold']} days ===")
        print(f"F1 Score: {row['F1 Score']:.3f}")
        print(f"Precision: {row['Precision']:.3f}")
        print(f"Recall: {row['Recall']:.3f}")
        print(f"Balanced Accuracy: {row['Balanced Accuracy']:.3f}")
        print(f"ROC AUC: {row['ROC AUC'] if row['ROC AUC'] is not None else 'N/A'}")
        print(f"Log Loss: {row['Log Loss'] if row['Log Loss'] is not None else 'N/A'}")
        print(f"RMSE (Overall): {row['RMSE (Overall)']:.3f}")
        print(f"MAE (Overall): {row['MAE (Overall)']:.3f}")
        print(f"R2 Score: {row['R2 Score']:.3f}")
        print(f"RMSE (Short Stay): {row['RMSE (Short Stay)'] if pd.notnull(row['RMSE (Short Stay)']) else 'N/A'}")
        print(f"RMSE (Long Stay): {row['RMSE (Long Stay)'] if pd.notnull(row['RMSE (Long Stay)']) else 'N/A'}")


# results_df = compare_los_thresholds(X, y_los_days, thresholds=[7,14,30], apply_smote=True, model_cfg=config.DEFAULT_MODEL_CONFIG)
# compare_los_results(results_df)



# LOSModel subclass

# class LOSModel(PatientOutcomeModel):
#     target_column = 'los_processed'

#     def preprocess_target(self):
#         """
#         Specific preprocessing for length of stay (LOS) target.
#         E.g. classification of short/medium/long stays,
#         capping values, or regression.
#         """
#         # Example:
#         # self.df['los_processed'] = preprocess_los(self.df['los'])
#         pass

#     def train_model(self):
#         # Implement your LOS model training here
#         pass

#     def evaluate(self):
#         # Evaluate LOS prediction
#         pass
