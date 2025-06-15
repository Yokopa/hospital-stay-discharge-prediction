import logging

import pandas as pd
from models.discharge_type_model import DischargeTypeModel
from models.los_model import LOSModel
import argparse
import pandas as pd
import config  # Assuming you have a config file
from data_preparation.feature_preprocessing import discharge_type_preprocess
#from your_module import clean_data, preprocess_data, preprocess_target, run_discharge_type_model, run_los_model  # replace with your real imports


def clean_data():
    from src.data_preparation.clean_raw_data import clean_lab_data
    from src.data_preparation.clean_raw_data import clean_clinical_data
    from data_preparation.integrate_data import integrate_data
    from data_preparation.clean_merged_data import replace_invalid_values, filter_adults
    from utils import generate_summary_statistics
    import config

    # Clean individual datasets
    df_clinical = clean_clinical_data(save=True, verbose=True)
    df_lab = clean_lab_data(save=True, verbose=True)

    # Merge lab + clinical
    df_merged = integrate_data(df_clinical, df_lab, save=True, verbose=True)

    # Replace invalids and filter to adults
    df_cleaned = replace_invalid_values(df_merged, save=True, verbose=True)
    df_cleaned = filter_adults(df_cleaned, save=True, verbose=True)

    # Summary
    generate_summary_statistics(
        config.CLEANED_MERGED_DATA_PATH,
        start_col=4,
        save_path=config.LAB_TEST_STATISTICS,
        verbose=True
    )

    return df_cleaned


def preprocess_data(target,  # 'discharge_type' or 'length_of_stay_days'
                    df=None,
                    missing_data_flags=False,
                    binning_age=False,
                    icd_codes='categories',  # default to 'categories'
                    encoding=False,
                    imputation=False,
                    scaling=False,
                    ordinal_cols=None,
                    ordinal_categories=None):
    """
    Preprocesses data: encoding, imputation, scaling, etc.
    
    Parameters:
        df (pd.DataFrame): Raw data
        target (str): target column name
        missing_data_flags (bool): whether to add missingness indicators
        binning_age (bool): whether to bin age into categories
        icd_codes (str): 'blocks', 'categories' or None
        encoding (bool): whether to encode categorical features
        imputation (bool): whether to impute missing values
        scaling (bool): whether to scale features
        ordinal_cols (list): list of ordinal columns for encoding
        ordinal_categories (dict): dict of categories for ordinal encoding
    
    Returns:
        processed DataFrames or dict with processed data (you can adjust)
    """

    # Import functions inside
    from data_preparation.feature_preprocessing import (
        add_missing_indicators, categorize_age, group_icd_to_blocks, categorize_icd,
        fit_encoders, transform_with_encoders, scale_features,
        impute_with_random_forest, clip_imputed_values
    )
    from data_preparation.split import train_test_split_by_patient
    import config

    # Check logic constraint: if imputation=True, encoding must be True
    if imputation and not encoding:
        raise ValueError("Encoding must be True if imputation is True.")

    # Start with clean data (or pass df if you want to allow that)
    if df is None:
        df = clean_data()

    # Add missingness flags
    if missing_data_flags:
        df = add_missing_indicators(df)

    # Bin age if requested
    if binning_age:
        df['age'] = categorize_age(df['age'])

    # Process ICD codes
    if icd_codes == 'categories':
        df = categorize_icd(df)
    elif icd_codes == 'blocks':
        df = group_icd_to_blocks(df)

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split_by_patient(
        df,
        target_col=target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=False
    )

    # Encoding if requested
    if encoding:
        # You need to provide or define ordinal_cols and ordinal_categories or set defaults
        if ordinal_cols is None:
            ordinal_cols = []
        if ordinal_categories is None:
            ordinal_categories = {}

        encoders = fit_encoders(X_train, ordinal_cols=ordinal_cols, ordinal_categories=ordinal_categories)
        X_train = transform_with_encoders(X_train, encoders)
        X_test = transform_with_encoders(X_test, encoders)

    # Imputation if requested (note: you should pass numeric DataFrames only here)
    if imputation:
        # Select numeric columns for imputation
        train_num_df = X_train.select_dtypes(include='number')
        test_num_df = X_test.select_dtypes(include='number')

        imputed_train, imputed_test = impute_with_random_forest(train_num_df, test_num_df)
        imputed_train = clip_imputed_values(train_num_df, imputed_train)
        imputed_test = clip_imputed_values(test_num_df, imputed_test)

        # Replace numeric columns with imputed values
        X_train.loc[imputed_train.index, imputed_train.columns] = imputed_train
        X_test.loc[imputed_test.index, imputed_test.columns] = imputed_test

    # Scaling if requested
    if scaling:
        X_train, X_test = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to raw dataset CSV')
    parser.add_argument('--target', type=str, required=True, choices=[config.DISCHARGE_TARGET, config.LOS_TARGET])
    parser.add_argument('--model', type=str, required=True, choices=['decision_tree', 'random_forest', 'xgboost', 'lightgbm', 'catboost'])

    args = parser.parse_args()



# LOAD
    lab_data = load_csv(config.LAB_DATA_PATH)
    cleaned_lab_data = clean_lab_data(lab_data)

    clin_data = load_csv(config.CLINICAL_DATA_PATH)
    cleaned_clin_data = clean_clinical_data(clin_data)

    # Preprocess features and split into train/test with target
    X_train, X_test, y_train, y_test = preprocess_data(
        df_cleaned,
        target=args.target,
        missing_data_flags=True,
        binning_age=True,
        icd_codes='categories',
        encoding=True,
        imputation=True,
        scaling=True
    )

    # Run the appropriate model function
    if args.target == config.DISCHARGE_TARGET:
        # Step 3: Preprocess target columns (apply label mapping etc.)
        y_train = discharge_type_preprocess(y_train, args.target, mode='train')
        y_test = discharge_type_preprocess(y_test, args.target, mode='test')

        run_discharge_type_model(X_train, y_train, X_test, y_test)

    elif args.target == config.LOS_TARGET:
        run_los_model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
