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
    clf_class = model_cfg[model_name]["classifier"]["class"]
    clf_params = model_cfg[model_name]["classifier"]["params"].copy()

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    # --- Handle imbalance for XGBoost (no SMOTE) ---
    if clf_class.__name__ == "XGBClassifier":
        clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train)

    # --- Pass categorical features if LightGBM ---
    if clf_class.__name__ == "LGBMClassifier" and categorical_features:
        clf_params["categorical_feature"] = categorical_features

    # --- Train model ---
    if clf_class.__name__ == "CatBoostClassifier":
        from catboost import Pool
        cat_indices = [X_train.columns.get_loc(col) for col in categorical_features]
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        test_pool = Pool(X_test, y_test, cat_features=cat_indices)
        classifier = clf_class(**clf_params)
        classifier.fit(train_pool, eval_set=test_pool, verbose=0)
        y_pred = classifier.predict(test_pool)
    else:
        classifier = clf_class(**clf_params)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

    # --- Metrics ---
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    roc_auc, logloss = None, None
    try:
        if clf_class.__name__ == "CatBoostClassifier":
            y_proba = classifier.predict_proba(test_pool)
        elif hasattr(classifier, "predict_proba"):
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

# def train_discharge_pipeline(
#     X_train, 
#     X_test, 
#     y_train, 
#     y_test, 
#     apply_smote, 
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
#         apply_smote: bool
#             Whether to apply SMOTE for class balancing.
#         model_name: str
#             Model key (e.g., 'xgboost', 'catboost').
#         model_cfg: dict
#             Configuration for model class and parameters.

#     Returns:
#         dict: Contains trained model and classification metrics.
#     """
#     clf_class = model_cfg[model_name]["classifier"]["class"]
#     clf_params = model_cfg[model_name]["classifier"]["params"].copy()

#     categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

#     # --- Apply SMOTE if specified ---
#     if apply_smote:
#         sm = SMOTE(random_state=config.RANDOM_SEED)
#         X_train, y_train = sm.fit_resample(X_train, y_train)
#         clf_params.pop("class_weight", None)
#         clf_params.pop("auto_class_weights", None)
#         clf_params.pop("is_unbalance", None)
#         clf_params.pop("scale_pos_weight", None)

#     # --- Handle imbalance for XGBoost ---
#     if not apply_smote and clf_class.__name__ == "XGBClassifier":
#         clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train)

#     # --- Pass categorical features if LightGBM ---
#     if clf_class.__name__ == "LGBMClassifier" and categorical_features:
#         clf_params["categorical_feature"] = categorical_features

#     # --- Train model ---
#     if clf_class.__name__ == "CatBoostClassifier":
#         from catboost import Pool
#         cat_indices = [X_train.columns.get_loc(col) for col in categorical_features]
#         train_pool = Pool(X_train, y_train, cat_features=cat_indices)
#         test_pool = Pool(X_test, y_test, cat_features=cat_indices)
#         classifier = clf_class(**clf_params)
#         classifier.fit(train_pool, eval_set=test_pool, verbose=0)
#         y_pred = classifier.predict(test_pool)
#     else:
#         classifier = clf_class(**clf_params)
#         classifier.fit(X_train, y_train)
#         y_pred = classifier.predict(X_test)

#     # --- Metrics ---
#     precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
#     cm = confusion_matrix(y_test, y_pred)
#     balanced_acc = balanced_accuracy_score(y_test, y_pred)

#     roc_auc, logloss = None, None
#     try:
#         if clf_class.__name__ == "CatBoostClassifier":
#             y_proba = classifier.predict_proba(test_pool)
#         elif hasattr(classifier, "predict_proba"):
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
