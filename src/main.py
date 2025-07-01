import sys
import os
import argparse
import logging
import datetime
from pathlib import Path

import pandas as pd
import joblib

# Add parent directory to sys.path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils
import config
import data_loader

from src.data_preparation import feature_preprocessing
from src.modeling import los_model, discharge_type_model

def main():
    # ---------------- ARGUMENT PARSING ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['clean_only', 'save_dataset_only', 'find_best_threshold', 'run_model'], required=True)
    parser.add_argument('--target', choices=['los', 'discharge_type'])  # only required for run_model
    parser.add_argument('--mode', choices=['regression', 'multiclass', 'two_step'])  # only required for los
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', choices=['lightgbm', 'catboost', 'xgboost', 'logistic_regression', 'linear_regression'])
    parser.add_argument('--param_set', type=str, default='default')
    parser.add_argument('--thresholds', type=int, nargs='+', default=None)
    parser.add_argument('--cleaned_data_path', type=str, default=config.CLEANED_MERGED_DATA_PATH)
    parser.add_argument('--save_dataset', action='store_true', help="Whether to save the prepared dataset.")
    parser.add_argument('--force_reprocess', action='store_true', help="Ignore cached dataset and reprocess it")
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_shap', action='store_true')
    parser.add_argument('--save_preds', action='store_true')
    args = parser.parse_args()

        # Force saving if the step is save_dataset_only
    if args.step == "save_dataset_only":
        args.save_dataset = True

    # ---------------- VALIDATION LOGIC ----------------
    if args.step == 'clean_only':
        pass  # nothing required

    elif args.step == 'save_dataset_only':
        required = [args.target, args.dataset_name]
        if not all(required):
            raise ValueError("--target and --dataset_name are required for save_dataset_only")

    elif args.step == 'find_best_threshold':
        # Only applies to LOS → target is implied
        if not all([args.dataset_name, args.model_name, args.param_set]):
            raise ValueError("--dataset_name, --model_name, and --param_set are required for find_best_threshold. --target is implied (los)")
        args.target = 'los'  # enforce target internally

    elif args.step == 'run_model':
        required = [args.target, args.dataset_name, args.model_name, args.param_set]
        if not all(required):
            raise ValueError("--target, --dataset_name, --model_name, and --param_set are required for run_model")
        if args.target == 'los':
            if not args.mode:
                raise ValueError("--mode is required when target is 'los'")
            if args.mode in ["multiclass", "two_step"]:
                if args.thresholds is None:
                    logging.info(f"No custom thresholds passed. Using default from config: {config.LOS_CLASSIFICATION_THRESHOLDS}")
                    thresholds = config.LOS_CLASSIFICATION_THRESHOLDS
                else:
                    logging.info(f"Using custom thresholds: {args.thresholds}")
                    thresholds = args.thresholds

    else:
        raise ValueError(f"Unsupported step: {args.step}")

    # ---------------- START STEP LOGGING ----------------
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Starting step: {args.step} at {now}")
    logging.info(f"Args: {vars(args)}")

    # ---------------- CUSTOM LOGGING ----------------
    if args.step == 'clean_only':
        logging.info("Step: clean_only - only data cleaning will be performed.")

    elif args.step == 'save_dataset_only':
        if args.target == "discharge_type":
            logging.info(f"Discharge categories: {config.DISCHARGE_CATEGORIES_NUMBER}")
        elif args.target == "los":
            logging.info(f"LOS transformation method: {config.LOS_TRANSFORMATION['method']}")

        logging.info(
            f"Step: {args.step} | Target: {args.target} | Dataset: {args.dataset_name} | "
            f"Saving prepared dataset only."
        )

    elif args.step == 'find_best_threshold':
        logging.info(
            f"Step: {args.step} | Target: los | Dataset: {args.dataset_name} | "
            f"Model: {args.model_name} | Param set: {args.param_set}"
        )

    elif args.step == 'run_model':
        if args.target == "discharge_type":
            logging.info(f"Discharge categories: {config.DISCHARGE_CATEGORIES_NUMBER}")
        elif args.target == "los":
            logging.info(f"LOS transformation method: {config.LOS_TRANSFORMATION['method']}")
        
        threshold_info = args.thresholds if args.target == 'los' and args.mode in ['multiclass', 'two_step'] else 'N/A'
        mode_info = args.mode if args.target == 'los' else 'N/A'

        logging.info(
            f"Step: {args.step} | Target: {args.target} | Mode: {mode_info} | "
            f"Dataset: {args.dataset_name} | Model: {args.model_name} | "
            f"Param set: {args.param_set} | Threshold(s): {threshold_info}"
        )
    
    logging.info("-" * 60)

    # ---------------- MODE: CLEAN ONLY ----------------
    if args.step == 'clean_only':
        cleaned = data_loader.load_or_clean_data(args.cleaned_data_path)
        logging.info(f"Cleaned data saved to {args.cleaned_data_path}")
        logging.info("-" * 60)
        return

    # ---------------- LOAD CONFIGS ----------------
    # At this point we need to load configs + data
    dataset_configs = utils.load_yaml(config.DATASET_CONFIG_PATH)
    model_configs = utils.load_yaml(config.MODEL_CONFIG_PATH)

    # ---------------- LOAD / PREPROCESS DATA ----------------
    # Always check dataset name
    if args.dataset_name not in dataset_configs:
        raise ValueError(f"Dataset config '{args.dataset_name}' not found in datasets.yaml")

    dataset_config = dataset_configs[args.dataset_name]  # Keep this — safe for all steps

    # Only check model config if needed for the step
    if args.step in ["find_best_threshold", "run_model"]:
        if args.model_name not in model_configs:
            raise ValueError(f"Model config '{args.model_name}' not found in models.yaml")
        if args.param_set not in model_configs[args.model_name]:
            raise ValueError(f"Param set '{args.param_set}' not found for model '{args.model_name}'")
        model_config = model_configs[args.model_name][args.param_set]
    else:
        model_config = None

    # ---------------- LOAD CLEANED DATA ----------------
    cleaned = data_loader.load_or_clean_data(args.cleaned_data_path)
    target_col = config.LOS_TARGET if args.target == "los" else config.DISCHARGE_TARGET
    
    dataset_save_dir = config.DATA_DIR / "processed"
    dataset_save_dir.mkdir(parents=True, exist_ok=True)
    # Optional suffixes
    discharge_suffix = f"{config.DISCHARGE_CATEGORIES_NUMBER}cat" if args.target == "discharge_type" else ""
    los_suffix = config.LOS_TRANSFORMATION["method"] if args.target == "los" else ""

    suffix = discharge_suffix or los_suffix or ""

    dataset_filename = f"{args.target}"
    if suffix:
        dataset_filename += f"_{suffix}"
    dataset_filename += f"_{args.dataset_name}_prepared.joblib"

    dataset_path = dataset_save_dir / dataset_filename


    # ---------------- LOAD OR PREPROCESS DATASET ----------------
    # Try to load preprocessed dataset if it exists
    if dataset_path.exists() and not args.force_reprocess:
        logging.info(f"Found existing dataset at {dataset_path}, loading it.")
        
        loaded = joblib.load(dataset_path)
        X_train = loaded["X_train"]
        X_test = loaded["X_test"]
        y_train = loaded["y_train"]
        y_test = loaded["y_test"]
        transformers = loaded["transformers"]

        # ---------------- MODE: SAVE DATASET ONLY ----------------
        if args.step == "save_dataset_only":
            logging.info("Step is save_dataset_only and dataset already exists. Skipping preprocessing.")
            return  # exit early

    else:
        logging.info("Dataset not found on disk. Running full preprocessing.")
        X_train, X_test, y_train, y_test, transformers = feature_preprocessing.prepare_dataset(
            raw_df=cleaned,
            target_col=target_col,
            dataset_config=dataset_config
        )

        # Save the preprocessed dataset (always True if save_dataset_only step)
        if args.save_dataset:
            joblib.dump({
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "transformers": transformers
            }, dataset_path)
            logging.info(f"Saved prepared dataset to {dataset_path}")
        else:
            logging.info("Dataset was not saved. Use --save_dataset to enable saving.")
            
        # Exit immediately after saving if the step was just to save the dataset
        if args.step == "save_dataset_only":
            logging.info("Step is save_dataset_only, exiting after dataset is saved.")
            return
        
        logging.info("-" * 60)
        
    # ---------------- STEP: FIND BEST THRESHOLD ----------------
    if args.step == "find_best_threshold":
        df_results = los_model.compare_los_thresholds( # CHOOSE WETHER TO USE 2-STEP PIPELINE OR JUST CLASSIFICATION
            X_train, X_test, y_train, y_test,
            threshold_sets=config.LOS_TARGET_THRESHOLDS,
            model_cfg=model_config,
            dataset_name=args.dataset_name
        )

        results_dir = config.RESULTS_DIR / "two_step_los" / "thresholds_comparison"
        results_dir.mkdir(parents=True, exist_ok=True)

        # out_path = results_dir / f"los_{args.model_name}_{args.param_set}_{args.dataset_name}_thresholds_{now}.csv"
        # df_results.to_csv(out_path, index=False)
        # logging.info(f"Saved threshold comparison results to {out_path}")
        df_results['dataset_name'] = args.dataset_name
        out_path = results_dir / f"los_{args.model_name}_{args.param_set}_thresholds.csv"

        if out_path.exists():
            df_results.to_csv(out_path, mode='a', header=False, index=False)
        else:
            df_results.to_csv(out_path, mode='w', header=True, index=False)

        logging.info(f"Saved/appended threshold comparison results to {out_path}")

    # ---------------- STEP: RUN MODEL ----------------
    elif args.step == "run_model":

        # ---------------- LOS ----------------
        if args.target == 'los':
            results_dir = config.RESULTS_DIR / f"{args.target}_{args.mode}"
            results_dir.mkdir(parents=True, exist_ok=True)

            if args.mode == "regression":
                result = los_model.train_los_regression_baseline(
                    X_train, X_test, y_train, y_test,
                    model_cfg=model_config
                )

            elif args.mode == "multiclass":
                los_bins = [0] + args.thresholds + [float("inf")]
                logging.info(f"Using LOS classification bins: {los_bins} (thresholds: {args.thresholds})")

                result = los_model.train_los_multiclass_baseline(
                    X_train, X_test, y_train, y_test,
                    bins=los_bins,
                    model_cfg=model_config
                )

            elif args.mode == "two_step":
                logging.info(f"Using thresholds for two-step modeling: {args.thresholds}")
                result = los_model.los_two_step_pipeline(
                    X_train, X_test, y_train, y_test,
                    thresholds=args.thresholds,           # pass the full list of thresholds
                    model_cfg=model_config
                )

            else:
                raise ValueError(f"Unsupported mode: {args.mode}")

        # ---------------- DISCHARGE_TYPE ----------------
        elif args.target == "discharge_type":
            results_dir = config.RESULTS_DIR / "discharge_type"
            results_dir.mkdir(parents=True, exist_ok=True)

            y_train_encoded, y_test_encoded, encoder = feature_preprocessing.encode_target(y_train, y_test)

            encoder_dir = config.MODELS_DIR / "discharge_type"
            encoder_dir.mkdir(parents=True, exist_ok=True)
            encoder_filename = f"label_encoder_{config.DISCHARGE_CATEGORIES_NUMBER}classes.joblib"
            encoder_path = encoder_dir / encoder_filename

            if not encoder_path.exists():
                joblib.dump(encoder, encoder_path)
                logging.info(f"Saved label encoder to {encoder_path}")
            else:
                logging.info(f"Label encoder already exists at {encoder_path}")

            result = discharge_type_model.train_discharge_pipeline(
                X_train,
                X_test,
                y_train_encoded,
                y_test_encoded,
                model_cfg=model_config,
            )

        # ---------------- SAVE RESULTS / MODELS ----------------
        # # Save results
        # df = pd.DataFrame([result])
        # out_path = results_dir / f"{args.target}_{args.model_name}_{args.param_set}_{args.dataset_name}_{now}.csv"
        # df.to_csv(out_path, index=False)
        # logging.info(f"Saved results to {out_path}")

        df = pd.DataFrame([result]) # All results for the same model and param_set across datasets go into one file
        df["dataset_name"] = args.dataset_name  # Add dataset column
        out_path = results_dir / f"{args.target}_{args.model_name}_{args.param_set}.csv"
        if out_path.exists():
            df.to_csv(out_path, mode='a', header=False, index=False)
        else:
            df.to_csv(out_path, mode='w', header=True, index=False)
        logging.info(f"Saved/appended results to {out_path}")

        # Save model(s)
        if args.save_model:
            models_dir = config.MODELS_DIR / args.target
            models_dir.mkdir(parents=True, exist_ok=True)
            model_prefix = f"{args.target}_{args.model_name}_{args.param_set}_{args.dataset_name}_{now}"

            if "classifier" in result:
                joblib.dump(result["classifier"], models_dir / f"{model_prefix}_classifier.joblib")
                logging.info(f"Saved classifier to {models_dir / f'{model_prefix}_classifier.joblib'}")

            if "regressor" in result:
                joblib.dump(result["regressor"], models_dir / f"{model_prefix}_regressor.joblib")
                logging.info(f"Saved regressor to {models_dir / f'{model_prefix}_regressor.joblib'}")

            if "confusion_matrix" in result:
                pd.DataFrame(result["confusion_matrix"]).to_csv(
                    results_dir / f"{model_prefix}_confusion_matrix.csv", index=False
                )
                logging.info(f"Saved confusion matrix to {results_dir / f'{model_prefix}_confusion_matrix.csv'}")

if __name__ == "__main__":
    utils.configure_logging(verbose=True)

    if len(sys.argv) == 1:
        sys.argv = [
            "main.py",
            "--step", "run_model",
            "--target", "los",
            "--dataset_name", "nan",
            "--model_name", "lightgbm",
            "--param_set", "default",
            "--mode", "two_step",
            "--thresholds", "7", "14", "30"
        ]
    main()
