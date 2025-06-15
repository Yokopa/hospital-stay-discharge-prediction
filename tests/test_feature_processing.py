# Tests for feature_preprocessing functions

import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from src.data_preparation.feature_preprocessing import *
from src import config

@pytest.fixture
def sample_data():
    data = {
        'patient_id': [1, 2, 3, 4, 1, 4, 5, 6],
        'case_id': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        'age': [25, 58, 70, np.nan, 26, 60, 80, 90],
        'diagnosis': ['I25.19', 'C34.1', 'E11.9', 'Z00.0', 'I25.19', 'Z00.0', 'C34.1', 'C34.1'],
        'feature1': [1.0, np.nan, 3.0, 4.0, 0, 2.0, 3.0, 0],
        'feature2': [np.nan, 2.0, 3.0, 4.0, 0, 4.0, 5.0, 2.0],
        'discharge_type': ['Home', 'Institution', 'Another hospital', 'Home', 'Deceased', 'Deceased','Institution', 'Another hospital'],
        'length_of_stay_days': [5, 10, 20, 30, 7, 2, 3, 1]
    }
    return pd.DataFrame(data)

def test_add_missing_indicators(sample_data):
    df = add_missing_indicators(sample_data.copy(), columns=['feature1', 'feature2'])
    assert 'feature1_missing' in df.columns
    assert 'feature2_missing' in df.columns
    assert df['feature1_missing'].iloc[1] == 1
    assert df['feature2_missing'].iloc[0] == 1

def test_categorize_age(sample_data):
    ages = sample_data['age']
    categorized = categorize_age(ages)
    expected = ['18-44', '45-59', '60-74', 'unknown', '18-44', '60-74', "75+", "75+"][:len(ages)]
    assert list(categorized) == expected

def test_group_icd_to_blocks(sample_data):
    df = group_icd_to_blocks(sample_data.copy(), column='diagnosis')
    assert df['diagnosis'].iloc[0] == 'I25'
    assert df['diagnosis'].nunique() == 4

def test_categorize_icd(sample_data):
    # Test single code
    code = 'C34.1'
    category = categorize_icd(code, icd_categories=config.ICD10_CATEGORIES)
    assert category == config.ICD10_CATEGORIES['II']
    # Test Series
    sample_data['icd_category'] = sample_data['diagnosis'].apply(categorize_icd)
    assert sample_data['icd_category'].nunique() == 4

def test_fit_transform_encoders(sample_data):
    df = sample_data.copy()
    df['age'] = categorize_age(df['age'].fillna(30))
    encoders = fit_encoders(df, ordinal_cols=['age'], ordinal_categories={'age': ['18-44', '45-59', '60-74', '75+']})
    transformed = transform_with_encoders(df, encoders)
    # Check for missing values only in the encoded 'age' column
    assert transformed['age'].isnull().sum() == 0
    # Check that the 'age' column is numeric
    assert pd.api.types.is_numeric_dtype(transformed['age'])
    # Check that the unique values are as expected
    assert set(transformed['age'].unique()).issubset({0, 1, 2, 0, 0, 2, 3, 3})

def test_scale_features(sample_data):
    X = sample_data[['feature1', 'feature2']].fillna(0)
    X_train, X_test = train_test_split(X, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

def test_impute_with_random_forest(sample_data):
    df = sample_data[['feature1', 'feature2']]
    train_df, test_df = df.iloc[:3], df.iloc[3:]
    imputed_train, imputed_test = impute_with_random_forest(train_df, test_df)
    assert not np.isnan(imputed_train.to_numpy()).any()
    assert not np.isnan(imputed_test.to_numpy()).any()
    assert imputed_train.shape == train_df.shape
    assert imputed_test.shape == test_df.shape

def test_clip_imputed_values(sample_data):
    original = sample_data[['feature1']]
    # Replace NaN with a high value
    imputed_high = original.fillna(original['feature1'].max() + 100)
    clipped_high = clip_imputed_values(original, imputed_high)
    assert clipped_high['feature1'].max() <= original['feature1'].max()
    # Replace NaN with a low value
    imputed_low = original.fillna(original['feature1'].min() - 100)
    clipped_low = clip_imputed_values(original, imputed_low)
    assert clipped_low['feature1'].min() >= original['feature1'].min()
    # Check shape is preserved
    assert clipped_high.shape == original.shape
    assert clipped_low.shape == original.shape

def test_train_test_split_by_patient(sample_data):
    df = sample_data.copy()
    X_train, X_test, y_train, y_test = train_test_split_by_patient(df, target_col=config.DISCHARGE_TARGET)
    # Check that patient_id and case_id are not in features
    assert set(X_train.columns).isdisjoint({'patient_id', 'case_id'})
    # Check that train and test indices do not overlap
    assert set(X_train.index).isdisjoint(X_test.index)
    # Optional: Check that each patient is only in one split
    train_patients = set(df.loc[X_train.index, 'patient_id'])
    test_patients = set(df.loc[X_test.index, 'patient_id'])
    assert train_patients.isdisjoint(test_patients)
    
def test_discharge_type_preprocess(sample_data):
    # Test 4 categories
    df_4 = discharge_type_preprocess(sample_data.copy(), mode='4_categories')
    assert df_4['discharge_type'].isnull().sum() == 0
    assert set(df_4['discharge_type'].unique()).issubset(set(config.DISCHARGE_TARGET_CATEGORIES_4))
    assert df_4['discharge_type'].nunique() <= 4
    # Test 3 categories
    df_3 = discharge_type_preprocess(sample_data.copy(), mode='3_categories')
    assert df_3['discharge_type'].isnull().sum() == 0
    assert set(df_3['discharge_type'].unique()).issubset(set(config.DISCHARGE_TARGET_CATEGORIES_3))
    assert df_3['discharge_type'].nunique() <= 3

def test_encode_decode_target(sample_data):
    X_train, X_test, y_train, y_test = train_test_split_by_patient(sample_data, target_col=config.DISCHARGE_TARGET)
    y_train_enc, y_test_enc, enc = encode_target(y_train, y_test)
    y_test_decoded = decode_target(y_test_enc, enc)
    assert y_test_enc.dtype.kind in {'i', 'u'}
    assert list(y_test_decoded) == list(y_test)

def test_prepare_dataset(sample_data):
    df = sample_data.copy()
    # Use one of predefined configs
    dataset_config = config.DATASET_CONFIGS_NAN_CAT_NATIVE[1]
    # If your prepare_dataset expects test_size and random_seed, pass them as needed
    X_train, X_test, y_train, y_test, _ = prepare_dataset(
        df,
        target_col=config.LOS_TARGET,
        dataset_config=dataset_config,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
        )
    assert not X_train.empty
    assert not y_test.empty
    assert not X_train.index.intersection(X_test.index).any()
    assert len(X_train) + len(X_test) == len(df)
