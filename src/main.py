import argparse
import pandas as pd
import logging
import sys
import config
import utils
from src.data_preparation import (
    clean_raw_data, integrate_data, feature_preprocessing
)
from models import los_model, discharge_type_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, choices=['los', 'discharge_type'])
    parser.add_argument('--dataset_config', type=str, required=True, choices=['nan_cat_native', 'nan_cat_encoded', 'no_nan_encoded'])
    parser.add_argument('--step', type=str, required=True, choices=['find_best_threshold', 'run_baseline', 'smote', 'xgboost'])
    parser.add_argument('--threshold', type=int, default=7, help="Threshold (for los only)")
    args = parser.parse_args()

    # Load target
    target = config.LOS_TARGET if args.target == 'los' else config.DISCHARGE_TARGET

    # Load dataset config
    dataset_config_dict = {
        'nan_cat_native': config.DATASET_CONFIGS_NAN_CAT_NATIVE,
        'nan_cat_encoded': config.DATASET_CONFIGS_NAN_CAT_ENCODED,
        'no_nan_encoded': config.DATASET_CONFIGS_NO_NAN_ENCODED
    }
    dataset_configs = dataset_config_dict[args.dataset_config]

    # Load and preprocess raw data
    lab_data = utils.load_csv(config.LAB_DATA_PATH)
    cleaned_lab_data = clean_raw_data.clean_lab_data(lab_data)
    clin_data = utils.load_csv(config.CLINICAL_DATA_PATH)
    cleaned_clin_data = clean_raw_data.clean_clinical_data(clin_data)
    merged_data = integrate_data.integrate_data(cleaned_clin_data, cleaned_lab_data)
    merged_data = integrate_data.replace_invalid_values(merged_data)
    cleaned_merged_data = integrate_data.filter_adults(
        df=merged_data,
        save=True,
        save_path=config.CLEANED_MERGED_DATA_PATH,
        verbose=True
    )
    utils.generate_summary_statistics(cleaned_merged_data, start_col=4, save_path=config.LAB_TEST_STATISTICS)

    # Prepare dataset versions and print info
    all_data = feature_preprocessing.generate_all_datasets(cleaned_merged_data, target, dataset_configs)

    for config_name, data in all_data.items():
        print(f"\n=== Dataset: {config_name} ===")
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        print("\nX_train head():")
        print(X_train.head())
        print("\ny_train head():")
        print(y_train.head())
        print("-" * 50)

        transformers = data["transformers"]
        print("\nTransformers used:")
        for k, v in transformers.items():
            print(f"  - {k}: {type(v).__name__ if not isinstance(v, str) else v}")

    # === LOS TARGET ===
    if args.target == 'los':
        if args.step == 'find_best_threshold':
            for name, data in all_data.items():
                print(f"\n=== Finding best threshold for dataset: {name} ===")
                df_results = los_model.compare_los_thresholds(
                    X=data["X_train"],
                    X_test=data["X_test"],
                    y_train=data["y_train"],
                    y_test=data["y_test"],
                    thresholds=[7, 14, 30],
                    apply_smote=False,
                    model_name="lightgbm",
                    model_cfg=config.MODEL_CONFIGS
                )
                los_model.display_thresholds_comparison(df_results)

        elif args.step == 'run_baseline':
            logging.info("Running baseline models on all dataset configurations...")
            results = []
            for config_ in dataset_configs:
                for model_name in ["lightgbm", "catboost", "xgboost"]:
                    logging.info(f"Training {model_name} on dataset: {config_.name}")
                    result = los_model.run_model_on_dataset(
                        raw_df=cleaned_merged_data,
                        target_col=target,
                        dataset_config=config_,
                        model_name=model_name,
                        model_cfg=config.MODEL_CONFIGS,
                        threshold=args.threshold,
                        apply_smote=False
                    )
                    results.append(result)
            df = pd.DataFrame(results)
            df.to_csv("baseline_results.csv", index=False)
            logging.info("Saved results to baseline_results.csv")

        elif args.step == 'smote':
            results = []
            for config_ in dataset_configs:
                for model_name in ["lightgbm", "catboost", "xgboost"]:
                    result = los_model.run_model_on_dataset(
                        raw_df=cleaned_merged_data,
                        target_col=target,
                        dataset_config=config_,
                        model_name=model_name,
                        model_cfg=config.MODEL_CONFIGS,
                        threshold=args.threshold,
                        apply_smote=True
                    )
                    results.append(result)
            df = pd.DataFrame(results)
            print(df)

        elif args.step == 'xgboost':
            results = []
            for config_ in dataset_configs:
                result = los_model.run_model_on_dataset(
                    raw_df=cleaned_merged_data,
                    target_col=target,
                    dataset_config=config_,
                    model_name="xgboost",
                    model_cfg=config.MODEL_CONFIGS,
                    threshold=args.threshold,
                    apply_smote=True
                )
                results.append(result)
            df = pd.DataFrame(results)
            print(df)

    # === DISCHARGE TYPE TARGET ===
    if args.target == 'discharge_type':
        if args.step == 'run_baseline':
            results = []
            for config_ in dataset_configs:
                for model_name in ["lightgbm", "catboost", "xgboost"]:
                    result = discharge_type_model.run_model_on_dataset(
                        raw_df=cleaned_merged_data,
                        target_col=target,
                        dataset_config=config_,
                        model_name=model_name,
                        model_cfg=config.MODEL_CONFIGS,
                        apply_smote=False
                    )
                    results.append(result)
            df = pd.DataFrame(results)
            df.to_csv("discharge_baseline_results.csv", index=False)
            logging.info("Saved results to discharge_baseline_results.csv")

        elif args.step == 'smote':
            results = []
            for config_ in dataset_configs:
                for model_name in ["lightgbm", "catboost", "xgboost"]:
                    result = discharge_type_model.run_model_on_dataset(
                        raw_df=cleaned_merged_data,
                        target_col=target,
                        dataset_config=config_,
                        model_name=model_name,
                        model_cfg=config.MODEL_CONFIGS,
                        apply_smote=True
                    )
                    results.append(result)
            df = pd.DataFrame(results)
            print(df)

        elif args.step == 'xgboost':
            results = []
            for config_ in dataset_configs:
                result = discharge_type_model.run_model_on_dataset(
                    raw_df=cleaned_merged_data,
                    target_col=target,
                    dataset_config=config_,
                    model_name="xgboost",
                    model_cfg=config.MODEL_CONFIGS,
                    apply_smote=True
                )
                results.append(result)
            df = pd.DataFrame(results)
            print(df)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
   
    if len(sys.argv) == 1:  # No CLI args provided
        sys.argv = [
            "main.py",
            "--target", "los",
            "--dataset_config", "nan_cat_native",
            "--step", "find_best_threshold"
        ]
        
    main()