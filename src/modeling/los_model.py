from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    balanced_accuracy_score, roc_auc_score, log_loss,
    mean_absolute_error, r2_score, root_mean_squared_error,
    f1_score, precision_score, recall_score
)
import pandas as pd
import numpy as np
import config
import utils

def train_los_regression_baseline(X_train, X_test, y_train, y_test, model_name, model_cfg):
    reg_class = model_cfg[model_name]["regressor"]["class"]
    reg_params = model_cfg[model_name]["regressor"]["params"].copy()
    reg_params["random_state"] = config.RANDOM_SEED

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

def train_los_multiclass_baseline(X_train, X_test, y_train, y_test, bins, model_name, model_cfg):
    y_train_cls = pd.cut(y_train, bins=bins, labels=False, right=False)
    y_test_cls = pd.cut(y_test, bins=bins, labels=False, right=False)

    clf_class = model_cfg[model_name]["classifier"]["class"]
    clf_params = model_cfg[model_name]["classifier"]["params"].copy()
    clf_params["random_state"] = config.RANDOM_SEED

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    if clf_class.__name__ == "CatBoostClassifier":
        from catboost import Pool
        cat_indices = [X_train.columns.get_loc(col) for col in categorical_features]
        train_pool = Pool(X_train, y_train_cls, cat_features=cat_indices)
        test_pool = Pool(X_test, y_test_cls, cat_features=cat_indices)
        classifier = clf_class(**clf_params)
        classifier.fit(train_pool, eval_set=test_pool, verbose=0)
        y_pred_cls = classifier.predict(test_pool)
    elif clf_class.__name__ == "LGBMClassifier":
        classifier = clf_class(**clf_params)
        classifier.fit(X_train, y_train_cls, categorical_feature=categorical_features)
        y_pred_cls = classifier.predict(X_test)
    else:
        classifier = clf_class(**clf_params)
        classifier.fit(X_train, y_train_cls)
        y_pred_cls = classifier.predict(X_test)

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

def train_los_pipeline(
    X_train, X_test, y_train, y_test, threshold, model_name, model_cfg
):
    y_train_cls = (y_train > threshold).astype(int)
    y_test_cls = (y_test > threshold).astype(int)

    X_train_cls, X_test_cls = X_train.copy(), X_test.copy()

    categorical_features = X_train_cls.select_dtypes(include=["category", "object"]).columns.tolist()

    clf_class = model_cfg[model_name]["classifier"]["class"]
    clf_params = model_cfg[model_name]["classifier"]["params"].copy()
    clf_params["random_state"] = config.RANDOM_SEED

    if clf_class.__name__ == "XGBClassifier":
        clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train_cls)

    if clf_class.__name__ == "CatBoostClassifier":
        from catboost import Pool
        cat_indices = [X_train_cls.columns.get_loc(col) for col in categorical_features]
        train_pool = Pool(X_train_cls, y_train_cls, cat_features=cat_indices)
        test_pool = Pool(X_test_cls, y_test_cls, cat_features=cat_indices)
        classifier = clf_class(**clf_params)
        classifier.fit(train_pool, eval_set=test_pool, verbose=0)
        y_pred_cls = classifier.predict(test_pool)
    elif clf_class.__name__ == "LGBMClassifier":
        classifier = clf_class(**clf_params)
        classifier.fit(X_train_cls, y_train_cls, categorical_feature=categorical_features)
        y_pred_cls = classifier.predict(X_test_cls)
    else:
        classifier = clf_class(**clf_params)
        classifier.fit(X_train_cls, y_train_cls)
        y_pred_cls = classifier.predict(X_test_cls)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test_cls, y_pred_cls, average='weighted')
    cm = confusion_matrix(y_test_cls, y_pred_cls)
    balanced_acc = balanced_accuracy_score(y_test_cls, y_pred_cls)

    roc_auc = None
    logloss = None
    try:
        if clf_class.__name__ == "CatBoostClassifier":
            y_pred_proba = classifier.predict_proba(test_pool)
        elif hasattr(classifier, "predict_proba"):
            y_pred_proba = classifier.predict_proba(X_test_cls)
        else:
            y_pred_proba = None

        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test_cls, y_pred_proba[:, 1])
            logloss = log_loss(y_test_cls, y_pred_proba)
    except Exception as e:
        print(f"[Warning] ROC AUC or Log Loss computation failed: {e}")

    reg_class = model_cfg[model_name]["regressor"]["class"]
    reg_params = model_cfg[model_name]["regressor"]["params"].copy()
    reg_params["random_state"] = config.RANDOM_SEED

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

def compare_los_thresholds(prepared_datasets, thresholds=[7, 14, 30], model_cfg=None, model_name="lightgbm"):
    results = []
    for dataset_name, data in prepared_datasets.items():
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        for th in thresholds:
            result = train_los_pipeline(
                X_train, X_test, y_train, y_test,
                threshold=th,
                model_name=model_name,
                model_cfg=model_cfg
            )
            results.append({
                "Dataset": dataset_name,
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

def compare_models_on_dataset(dataset_name, data, model_cfg, model_names, threshold=14):
    """
    Compare different models on a single dataset.
    """
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    results = []

    for model_name in model_names:
        result = train_los_pipeline(
            X_train, X_test, y_train, y_test,
            threshold=threshold,
            model_name=model_name,
            model_cfg=model_cfg
        )
        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
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

def compare_models_across_datasets(prepared_datasets, model_cfg, model_names, threshold=14):
    """
    Run comparisons for all models across all datasets at a fixed threshold.
    """
    all_results = []
    for dataset_name, data in prepared_datasets.items():
        df = compare_models_on_dataset(dataset_name, data, model_cfg, model_names, threshold)
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True)

import os

def display_comparison_table(
    results_df,
    index_col="Model",
    dataset_params=None,
    model_params=None,
    save=False,
    save_dir="comparison_outputs"
):
    """
    Display and optionally save comparison metrics tables, highlighting best results per dataset.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain 'Dataset', index_col, and metric columns.

    index_col : str, default='Model'
        Column to use as the index (e.g., "Model" or "Threshold").

    dataset_params : dict or None
        Optional dictionary with dataset parameter info for saving.

    model_params : dict or None
        Optional dictionary with model parameter info for saving.

    save : bool, default=False
        If True, saves pivot tables and params to CSV files in save_dir.

    save_dir : str, default="comparison_outputs"
        Directory where CSV files will be saved.

    Behavior
    --------
    For each metric:
    - Prints a pivot table with rows=index_col, columns=Dataset.
    - Highlights the best value in each dataset (max for metrics like F1, min for errors like RMSE).
    - Optionally saves pivot tables and params CSVs with descriptive filenames.
    """

    import pandas as pd

    # Create dir if saving
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define metrics where lower is better
    lower_better = {"Log Loss", "RMSE (Overall)", "MAE (Overall)"}

    metrics = [
        "F1 Score", "Precision", "Recall", "Balanced Accuracy", "ROC AUC",
        "Log Loss", "RMSE (Overall)", "MAE (Overall)", "R2 Score"
    ]

    saved_files = []

    for metric in metrics:
        print(f"\n--- {metric} ---")
        if metric not in results_df.columns:
            print(f"{metric} not available in this results_df.")
            continue

        pivot = results_df.pivot(index=index_col, columns="Dataset", values=metric)

        # Highlight best: min or max per column
        def highlight_best(s):
            if metric in lower_better:
                is_best = s == s.min()
            else:
                is_best = s == s.max()
            return ['**' + f"{v:.3f}" + '**' if best else f"{v:.3f}" for v, best in zip(s, is_best)]

        # Print highlighted table
        highlighted = pivot.apply(highlight_best)
        for idx, row in highlighted.iterrows():
            print(f"{idx}: " + " | ".join(row))

        # Save pivot table csv
        if save:
            filename = f"{save_dir}/comparison_{index_col}_{metric.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
            pivot.round(5).to_csv(filename)
            saved_files.append(filename)

    # Save dataset and model params if provided and saving enabled
    if save:
        if dataset_params:
            ds_params_file = f"{save_dir}/dataset_params.csv"
            pd.DataFrame.from_dict(dataset_params, orient='index').to_csv(ds_params_file)
            saved_files.append(ds_params_file)
        if model_params:
            model_params_file = f"{save_dir}/model_params.csv"
            pd.DataFrame.from_dict(model_params, orient='index').to_csv(model_params_file)
            saved_files.append(model_params_file)

        print("\nSaved files:")
        for f in saved_files:
            print(f" - {f}")

# USage example:
# After generating results_df from model or threshold comparisons...
# display_comparison_table(
#     results_df,
#     index_col="Model",            # or "Threshold"
#     dataset_params={"ds1": {"desc": "Imputed data"}, "ds2": {"desc": "Raw data"}},
#     model_params={"xgboost": {"max_depth": 6}, "catboost": {"iterations": 1000}},
#     save=True,
#     save_dir="my_results"
# )



#-----------------------------
# WITH SMOTE (Consider removing it)
#-----------------------------
# def train_los_pipeline(
#         X_train, 
#         X_test, 
#         y_train, 
#         y_test, 
#         threshold, 
#         apply_smote, 
#         model_name, 
#         model_cfg):

#     """
#     Train a two-step Length of Stay (LOS) prediction pipeline:
#     1. Classification: predict whether LOS exceeds a given threshold.
#     2. Regression: predict the actual LOS values.

#     This function handles optional class balancing via SMOTE on the classification step,
#     trains specified models for classification and regression using provided config,
#     and returns various classification and regression performance metrics.

#     Args:
#         X_train : pd.DataFrame or np.ndarray
#             Training features.
#         X_test : pd.DataFrame or np.ndarray
#             Test features.
#         y_train : pd.Series or np.ndarray
#             Training target LOS values (continuous).
#         y_test : pd.Series or np.ndarray
#             Test target LOS values (continuous).
#         threshold : int or float
#             Threshold in days to convert LOS into a binary classification target.
#             Examples: 7, 14, 30 days.
#         apply_smote : bool
#             Whether to apply SMOTE oversampling to balance classes in classification.
#         model_name : str
#             Key to select model configuration (e.g., 'xgboost', 'catboost') from model_cfg.
#         model_cfg : dict
#             Dictionary containing model classes and parameters for classification and regression.

#     Returns:
#         dict
#             Dictionary containing trained classifier and regressor objects, classification metrics
#             (precision, recall, F1 score, confusion matrix, balanced accuracy, ROC AUC, log loss),
#             regression metrics (RMSE, MAE, R2), RMSE for short and long stay groups, and threshold used.
#     """

#     # --- Prepare classification target ---
#     y_train_cls = (y_train > threshold).astype(int)
#     y_test_cls = (y_test > threshold).astype(int)

#     X_train_cls, X_test_cls = X_train.copy(), X_test.copy()

#     # --- Detect categorical features ---
#     categorical_features = X_train_cls.select_dtypes(include=["category", "object"]).columns.tolist()

#     clf_class = model_cfg[model_name]["classifier"]["class"]
#     clf_params = model_cfg[model_name]["classifier"]["params"].copy()

#     # # --- Apply SMOTE if needed ---
#     # if apply_smote:
#     #     sm = SMOTE(random_state=config.RANDOM_SEED)
#     #     X_train_cls, y_train_cls = sm.fit_resample(X_train_cls, y_train_cls)
#     #     # Remove class balancing params if SMOTE is applied
#     #     clf_params.pop("class_weight", None)
#     #     clf_params.pop("auto_class_weights", None)
#     #     clf_params.pop("is_unbalance", None)
#     #     clf_params.pop("scale_pos_weight", None)
#     # else:
#     #     y_train_cls = y_train_cls

#     # --- Handle class imbalance for XGBoost ---
#     if not apply_smote and clf_class.__name__ == "XGBClassifier":
#         clf_params["scale_pos_weight"] = utils.compute_scale_pos_weight(y_train_cls)

#     # --- Train classifier ---
#     if clf_class.__name__ == "CatBoostClassifier":
#         from catboost import Pool
#         cat_indices = [X_train_cls.columns.get_loc(col) for col in categorical_features]
#         train_pool = Pool(X_train_cls, y_train_cls, cat_features=cat_indices)
#         test_pool = Pool(X_test_cls, y_test_cls, cat_features=cat_indices)
#         classifier = clf_class(**clf_params)
#         classifier.fit(train_pool, eval_set=test_pool, verbose=0)
#         y_pred_cls = classifier.predict(test_pool)

#     elif clf_class.__name__ == "LGBMClassifier":
#         classifier = clf_class(**clf_params)
#         classifier.fit(
#             X_train_cls,
#             y_train_cls,
#             categorical_feature=categorical_features
#         )
#         y_pred_cls = classifier.predict(X_test_cls)

#     else:
#         classifier = clf_class(**clf_params)
#         classifier.fit(X_train_cls, y_train_cls)
#         y_pred_cls = classifier.predict(X_test_cls)

#     # --- Classification metrics ---
#     precision, recall, f1, _ = precision_recall_fscore_support(y_test_cls, y_pred_cls, average='weighted')
#     cm = confusion_matrix(y_test_cls, y_pred_cls)
#     balanced_acc = balanced_accuracy_score(y_test_cls, y_pred_cls)

#     roc_auc = None
#     logloss = None
#     try:
#         if clf_class.__name__ == "CatBoostClassifier":
#             y_pred_proba = classifier.predict_proba(test_pool)
#         elif hasattr(classifier, "predict_proba"):
#             y_pred_proba = classifier.predict_proba(X_test_cls)
#         else:
#             y_pred_proba = None

#         if y_pred_proba is not None:
#             roc_auc = roc_auc_score(y_test_cls, y_pred_proba[:, 1])
#             logloss = log_loss(y_test_cls, y_pred_proba)
#     except Exception:
#         pass

#     # --- Regression step ---
#     reg_class = model_cfg[model_name]["regressor"]["class"]
#     reg_params = model_cfg[model_name]["regressor"]["params"].copy()
#     regressor = reg_class(**reg_params)
#     regressor.fit(X_train, y_train)
#     y_pred_reg = regressor.predict(X_test)

#     rmse = root_mean_squared_error(y_test, y_pred_reg)
#     mae = mean_absolute_error(y_test, y_pred_reg)
#     r2 = r2_score(y_test, y_pred_reg)

#     # --- RMSE for short and long stay groups ---
#     rmse_short, rmse_long = None, None
#     try:
#         short_mask = y_test_cls == 0
#         long_mask = y_test_cls == 1
#         if short_mask.any():
#             rmse_short = root_mean_squared_error(y_test[short_mask], y_pred_reg[short_mask])
#         if long_mask.any():
#             rmse_long = root_mean_squared_error(y_test[long_mask], y_pred_reg[long_mask])
#     except Exception:
#         pass

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

# def compare_los_thresholds(prepared_datasets, thresholds=[7, 14, 30], apply_smote=True, model_cfg=None, model_name="lightgbm"):
#     """
#     Run train_los_pipeline across all datasets and thresholds.
#     Returns a DataFrame of results.
#     """
#     results = []
#     for dataset_name, data in prepared_datasets.items():
#         X_train, X_test = data["X_train"], data["X_test"]
#         y_train, y_test = data["y_train"], data["y_test"]

#         for th in thresholds:
#             result = train_los_pipeline(
#                 X_train, X_test, y_train, y_test,
#                 threshold=th,
#                 apply_smote=apply_smote,
#                 model_name=model_name,
#                 model_cfg=model_cfg
#             )
#             results.append({
#                 "Dataset": dataset_name,
#                 "Threshold": th,
#                 "F1 Score": result["f1_score"],
#                 "Precision": result["precision"],
#                 "Recall": result["recall"],
#                 "Balanced Accuracy": result["balanced_accuracy"],
#                 "ROC AUC": result["roc_auc"],
#                 "Log Loss": result["log_loss"],
#                 "RMSE (Overall)": result["rmse"],
#                 "MAE (Overall)": result["mae"],
#                 "R2 Score": result["r2"],
#                 "RMSE (Short Stay)": result.get("rmse_short"),
#                 "RMSE (Long Stay)": result.get("rmse_long"),
#             })
#     return pd.DataFrame(results)

# def display_thresholds_comparison(results_df):
#     metrics = [
#         "F1 Score", "Precision", "Recall", "Balanced Accuracy", "ROC AUC",
#         "Log Loss", "RMSE (Overall)", "MAE (Overall)", "R2 Score"
#     ]
    
#     datasets = results_df["Dataset"].unique()
#     thresholds = sorted(results_df["Threshold"].unique())
    
#     for dataset in datasets:
#         print(f"\n=== Dataset: {dataset} ===")
#         subset = results_df[results_df["Dataset"] == dataset]
        
#         # Pivot for each metric with threshold columns
#         for metric in metrics:
#             pivot_table = subset.pivot(index="Dataset", columns="Threshold", values=metric)
#             print(f"\n{metric}:")
#             print(pivot_table.to_string(header=True))


# def display_thresholds_comparison(results_df):
#     metrics = [
#         "F1 Score", "Precision", "Recall", "Balanced Accuracy", "ROC AUC",
#         "Log Loss", "RMSE (Overall)", "MAE (Overall)", "R2 Score"
#     ]

#     datasets = results_df["Dataset"].unique()
#     thresholds = sorted(results_df["Threshold"].unique())

#     for dataset in datasets:
#         print(f"\n=== Dataset: {dataset} ===")
#         subset = results_df[results_df["Dataset"] == dataset]

#         for metric in metrics:
#             pivot_table = subset.pivot(index="Dataset", columns="Threshold", values=metric)
#             print(f"\n{metric}:")
#             print(pivot_table.to_string(header=True))


# Usage example (assuming you have raw_df, target_col, dataset_configs, model_cfg ready):
# prepared_datasets = generate_all_datasets(raw_df, target_col, dataset_configs)
# results_df = compare_los_thresholds(prepared_datasets, thresholds=[7,14,30], apply_smote=True, model_cfg=model_cfg, model_name="xgboost")
# compare_los_results(results_df)