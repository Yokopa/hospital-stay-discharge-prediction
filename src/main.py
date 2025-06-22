import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
import pandas as pd
import datetime
import utils
import config
import data_loader
from src.data_preparation import feature_preprocessing
from src.modeling import los_model, discharge_type_model
from pathlib import Path
from joblib import dump

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['los', 'discharge_type'], default='los')
    parser.add_argument('--dataset_name', type=str,
                        help="Name of dataset config as in datasets.yaml")
    parser.add_argument('--model_name', choices=['lightgbm', 'catboost', 'xgboost'])
    parser.add_argument('--param_set', type=str, default='default',
                        help="Parameter set name for model config, e.g. 'default', 'fast_train'")
    parser.add_argument('--step', choices=['clean_only','find_best_threshold', 'baseline_los', 'run_model'], required=True)
    parser.add_argument('--threshold', type=int, default=7, help="LOS threshold, if applicable")
    parser.add_argument('--cleaned_data_path', type=str, default=config.CLEANED_MERGED_DATA_PATH)
    parser.add_argument('--save_model', action='store_true', help="Whether to save trained model(s)")
    args = parser.parse_args()

    # ------------- VALIDATION BASED ON STEP ------------
    if args.step == 'clean_only':
        pass  # no validation needed
    elif args.step in ['baseline_los', 'find_best_threshold']:
        if args.target != 'los':
            raise ValueError(f"Step '{args.step}' only supports target='los'")
        if not args.dataset_name or not args.model_name:
            raise ValueError(f"--dataset_name and --model_name required for step '{args.step}'")
    elif args.step == 'run_model':
        if not args.target or not args.dataset_name or not args.model_name:
            raise ValueError("--target, --dataset_name, and --model_name required for 'run_model' step")

    # ------------- CUSTOM LOGGING ------------
    if args.step == 'clean_only':
        logging.info("Step: clean_only - only data cleaning will be performed.")
    elif args.step == 'find_best_threshold':
        logging.info(f"Step: {args.step} | Target: {args.target} | Dataset: {args.dataset_name} | "
                     f"Model: {args.model_name} | Param set: {args.param_set}")
    elif args.step == 'baseline_los':
        logging.info(f"Step: {args.step} | Target: {args.target} | Dataset: {args.dataset_name} | "
                     f"Model: {args.model_name} | Param set: {args.param_set} | Threshold: {args.threshold}")
    elif args.step == 'run_model':
        logging.info(f"Step: {args.step} | Target: {args.target} | Dataset: {args.dataset_name} | "
                     f"Model: {args.model_name} | Param set: {args.param_set} | Threshold: {args.threshold if args.target == 'los' else 'N/A'}")

    # Compute current timestamp for result files
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean-only logic (no model/dataset config needed)
    if args.step == 'clean_only':
        cleaned_merged = data_loader.load_or_clean_data(args.cleaned_data_path)
        logging.info(f"Step: clean_only completed at {now}")
        logging.info(f"Cleaned data saved to {args.cleaned_data_path}")
        return  # exit early

    # ------------- LOAD CONFIGS ------------
    dataset_configs = utils.load_yaml(config.DATASET_CONFIG_PATH)
    model_configs = utils.load_yaml(config.MODEL_CONFIG_PATH)

    if args.dataset_name not in dataset_configs:
        raise ValueError(f"Dataset config '{args.dataset_name}' not found in datasets.yaml")

    if args.model_name not in model_configs:
        raise ValueError(f"Model config '{args.model_name}' not found in models.yaml")

    model_param_sets = model_configs[args.model_name]
    if args.param_set not in model_param_sets:
        raise ValueError(f"Param set '{args.param_set}' not found for model '{args.model_name}'")

    dataset_config = dataset_configs[args.dataset_name]
    model_config = model_param_sets[args.param_set]

    # ------------- DEBUG PRINTS ------------
    print("model_config keys:", model_config.keys())  # dict_keys(['classifier', 'regressor'])
    print("classifier class:", model_config["classifier"]["class"])

    # ------------- DATA PREP ------------
    cleaned_merged = data_loader.load_or_clean_data(args.cleaned_data_path)
    target_col = config.LOS_TARGET if args.target == 'los' else config.DISCHARGE_TARGET

    X_train, X_test, y_train, y_test, transformers = feature_preprocessing.prepare_dataset(
        raw_df=cleaned_merged,
        target_col=target_col,
        dataset_config=dataset_config
    )

    if args.step == "baseline_los" and args.target == "los":

        # Run regression baseline
        regression_result = los_model.train_los_regression_baseline(
            X_train, X_test, y_train, y_test,
            model_cfg=model_config
        )

        # Run multiclass baseline
        los_bins = config.LOS_CLASSIFICATION_BINS
        multiclass_result = los_model.train_los_multiclass_baseline(
            X_train, X_test, y_train, y_test,
            bins=los_bins,
            model_cfg=model_config
        )

        # Build base directory
        results_dir = config.RESULTS_DIR / "baseline_los"
        (Path(results_dir / "regression")).mkdir(parents=True, exist_ok=True)
        (Path(results_dir / "multiclass")).mkdir(parents=True, exist_ok=True)
        # (Path(results_dir / "summary")).mkdir(parents=True, exist_ok=True)

        # Regression result
        reg_result = {
            "timestamp": now,
            "step": args.step,
            "model": args.model_name,
            "dataset": args.dataset_name,
            "target": args.target,
            "param_set": args.param_set,
            "type": "regression",
            "rmse": regression_result["rmse"],
            "mae": regression_result["mae"],
            "r2": regression_result["r2"]
        }
        reg_df = pd.DataFrame([reg_result])
        reg_path = results_dir / "regression" / f"{args.model_name}_{args.dataset_name}_{now}.csv"
        reg_df.to_csv(reg_path, index=False)

        # Multiclass classification result
        clf_result = {
            "timestamp": now,
            "step": args.step,
            "model": args.model_name,
            "dataset": args.dataset_name,
            "target": args.target,
            "param_set": args.param_set,
            "type": "multiclass",
            "multiclass_f1": multiclass_result["f1_score"],
            "multiclass_precision": multiclass_result["precision"],
            "multiclass_recall": multiclass_result["recall"],
            "multiclass_balanced_acc": multiclass_result["balanced_accuracy"],
            "multiclass_num_classes": multiclass_result["num_classes"]
        }
        clf_df = pd.DataFrame([clf_result])
        clf_path = results_dir / "multiclass" / f"{args.model_name}_{args.dataset_name}_{now}.csv"
        clf_df.to_csv(clf_path, index=False)

        # # Optional: merged summary row
        # summary_df = pd.DataFrame([{**reg_result, **clf_result}])
        # summary_path = results_dir / "summary" / f"{args.model_name}_{args.dataset_name}_{now}.csv"
        # summary_df.to_csv(summary_path, index=False)

        # # Optional: append to master log
        # master_log = results_dir / "all_runs_summary.csv"
        # summary_df.to_csv(master_log, mode="a", header=not master_log.exists(), index=False)

        logging.info(f"Saved regression to {reg_path}")
        logging.info(f"Saved multiclass to {clf_path}")
        # logging.info(f"Saved combined summary to {summary_path}")

    elif args.step == "find_best_threshold" and args.target == 'los':
        df_results = los_model.compare_los_thresholds(
            X_train, X_test, y_train, y_test,
            model_name=args.model_name,
            model_cfg=model_config,
            dataset_name=args.dataset_name,
            thresholds=config.LOS_TARGET_THRESHOLDS
        )

        # Build base directory
        results_dir = config.RESULTS_DIR / "two_step_los"
        (Path(results_dir / "thresholds_comparison")).mkdir(parents=True, exist_ok=True)
        out_path = f"{results_dir}/thresholds_comparison/{args.target}_{args.model_name}_{args.param_set}_{args.dataset_name}_thresholds_{now}.csv"
        df_results.to_csv(out_path, index=False)
        logging.info(f"Saved threshold comparison results to {out_path}")

        # # # Save pivot summaries
        # utils.display_comparison_table(
        #     df_results,
        #     index_col="Threshold",
        #     dataset_params={"dataset_name": args.dataset_name},
        #     model_params={"model_name": args.model_name, "param_set": args.param_set},
        #     save=True,
        #     save_dir=f"{config.RESULTS_DIR}/two_step_los/thresholds_comparison"
        # )

    elif args.step == "run_model":
        if args.target == 'los':
            result = los_model.los_two_step_pipeline(
                X_train, X_test, y_train, y_test,
                threshold=args.threshold,
                model_name=args.model_name,
                model_cfg=model_config
            )
            results_dir = config.RESULTS_DIR / "two_step_los"
        elif args.target == "discharge_type":
            result = discharge_type_model.train_discharge_pipeline( # adjust this function!
                X_train, 
                X_test, 
                y_train, 
                y_test, 
                model_name=args.model_name, 
                model_cfg=model_config,
            )
            results_dir = config.RESULTS_DIR / "discharge_type"

        # Create the results directory if it doesn't exist
        results_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([result])
        out_path = f"{results_dir}/{args.target}_{args.model_name}_{args.param_set}_{args.dataset_name}_results_{now}.csv"
        df.to_csv(out_path, index=False)
        logging.info(f"Saved results to {out_path}")

        if args.save_model:
            models_dir = config.MODELS_DIR / args.target
            models_dir.mkdir(parents=True, exist_ok=True)

            model_prefix = f"{args.target}_{args.model_name}_{args.param_set}_{args.dataset_name}_{now}"

            # Save classifier
            if "classifier" in result:
                clf_path = models_dir / f"{model_prefix}_classifier.joblib"
                dump(result["classifier"], clf_path)
                logging.info(f"Saved classifier to {clf_path}")

            # Save regressor if present (e.g., in LOS two-step)
            if "regressor" in result:
                reg_path = models_dir / f"{model_prefix}_regressor.joblib"
                dump(result["regressor"], reg_path)
                logging.info(f"Saved regressor to {reg_path}")

            # Save confusion matrix if present
            if "confusion_matrix" in result:
                cm_path = results_dir / f"{model_prefix}_confusion_matrix.csv"
                pd.DataFrame(result["confusion_matrix"]).to_csv(cm_path, index=False)
                logging.info(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Only set default args if none are provided (len(sys.argv) == 1 means only script name)
    if len(sys.argv) == 1:
        sys.argv = [
            "main.py",
            "--target", "los",
            "--dataset_name", "nan",
            "--model_name", "lightgbm",
            "--step", "baseline_los"
        ]

    main()
