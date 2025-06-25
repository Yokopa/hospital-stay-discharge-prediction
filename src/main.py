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
from joblib import dump

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['clean_only', 'find_best_threshold', 'run_model'], required=True)
    parser.add_argument('--target', choices=['los', 'discharge_type'])  # only required for run_model
    parser.add_argument('--mode', choices=['regression', 'multiclass', 'two_step'])  # only required for los
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', choices=['lightgbm', 'catboost', 'xgboost'])
    parser.add_argument('--param_set', type=str, default='default')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[7])
    parser.add_argument('--cleaned_data_path', type=str, default=config.CLEANED_MERGED_DATA_PATH)
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()

    # ---------------- VALIDATION LOGIC ----------------
    if args.step == 'clean_only':
        pass  # nothing required

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
            if args.mode in ['multiclass', 'two_step'] and not args.thresholds:
                raise ValueError("--thresholds required for multiclass or two_step LOS modeling")

    else:
        raise ValueError(f"Unsupported step: {args.step}")

    # ---------------- START STEP LOGGING ----------------
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Starting step: {args.step} at {now}")
    logging.info(f"Args: {vars(args)}")

    # ------------- CUSTOM LOGGING ------------
    if args.step == 'clean_only':
        logging.info("Step: clean_only - only data cleaning will be performed.")

    elif args.step == 'find_best_threshold':
        logging.info(
            f"Step: {args.step} | Target: los | Dataset: {args.dataset_name} | "
            f"Model: {args.model_name} | Param set: {args.param_set}"
        )

    elif args.step == 'run_model':
        threshold_info = args.thresholds if args.target == 'los' and args.mode in ['multiclass', 'two_step'] else 'N/A'
        mode_info = args.mode if args.target == 'los' else 'N/A'

        logging.info(
            f"Step: {args.step} | Target: {args.target} | Mode: {mode_info} | "
            f"Dataset: {args.dataset_name} | Model: {args.model_name} | "
            f"Param set: {args.param_set} | Threshold(s): {threshold_info}"
        )

    # ---------------- STEP: CLEAN ONLY ----------------
    if args.step == 'clean_only':
        cleaned = data_loader.load_or_clean_data(args.cleaned_data_path)
        logging.info(f"Cleaned data saved to {args.cleaned_data_path}")
        return

    # At this point we know we need to load configs + data
    dataset_configs = utils.load_yaml(config.DATASET_CONFIG_PATH)
    model_configs = utils.load_yaml(config.MODEL_CONFIG_PATH)

    if args.dataset_name not in dataset_configs:
        raise ValueError(f"Dataset config '{args.dataset_name}' not found in datasets.yaml")
    if args.model_name not in model_configs:
        raise ValueError(f"Model config '{args.model_name}' not found in models.yaml")
    if args.param_set not in model_configs[args.model_name]:
        raise ValueError(f"Param set '{args.param_set}' not found for model '{args.model_name}'")

    dataset_config = dataset_configs[args.dataset_name]
    model_config = model_configs[args.model_name][args.param_set]

    cleaned = data_loader.load_or_clean_data(args.cleaned_data_path)
    target_col = config.LOS_TARGET if args.target == "los" else config.DISCHARGE_TARGET

    X_train, X_test, y_train, y_test, transformers = feature_preprocessing.prepare_dataset(
        raw_df=cleaned,
        target_col=target_col,
        dataset_config=dataset_config
    )

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

    elif args.step == "run_model":
        if args.target == 'los':
            results_dir = config.RESULTS_DIR / f"{args.target}_{args.mode}"
            results_dir.mkdir(parents=True, exist_ok=True)

            if args.mode == "regression":
                result = los_model.train_los_regression_baseline(
                    X_train, X_test, y_train, y_test,
                    model_cfg=model_config
                )

            elif args.mode == "multiclass":
                los_bins = config.LOS_CLASSIFICATION_BINS
                result = los_model.train_los_multiclass_baseline(
                    X_train, X_test, y_train, y_test,
                    bins=los_bins,
                    model_cfg=model_config
                )

            elif args.mode == "two_step":
                result = los_model.los_two_step_pipeline(
                    X_train, X_test, y_train, y_test,
                    thresholds=args.thresholds,           # pass the full list of thresholds
                    model_cfg=model_config
                )

            else:
                raise ValueError(f"Unsupported mode: {args.mode}")

        elif args.target == "discharge_type":
            results_dir = config.RESULTS_DIR / "discharge_type"
            results_dir.mkdir(parents=True, exist_ok=True)

            y_train_encoded, y_test_encoded, encoder = feature_preprocessing.encode_target(y_train, y_test)

            encoder_dir = config.MODELS_DIR / "discharge_type"
            encoder_dir.mkdir(parents=True, exist_ok=True)
            encoder_filename = f"label_encoder_{config.DISCHARGE_CATEGORIES_NUMBER}classes.joblib"
            encoder_path = encoder_dir / encoder_filename

            if not encoder_path.exists():
                dump(encoder, encoder_path)
                logging.info(f"Saved label encoder to {encoder_path}")
            else:
                logging.info(f"Label encoder already exists at {encoder_path}")

            result = discharge_type_model.train_discharge_pipeline(
                X_train,
                X_test,
                y_train_encoded,
                y_test_encoded,
                model_name=args.model_name,
                model_cfg=model_config,
            )

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
                dump(result["classifier"], models_dir / f"{model_prefix}_classifier.joblib")
                logging.info(f"Saved classifier to {models_dir / f'{model_prefix}_classifier.joblib'}")

            if "regressor" in result:
                dump(result["regressor"], models_dir / f"{model_prefix}_regressor.joblib")
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
