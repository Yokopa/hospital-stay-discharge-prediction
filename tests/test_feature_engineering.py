# Tests for feature_engineering functions: anemia, kidney, and APRI

import pandas as pd
import numpy as np

from src.data_preparation.feature_engineering import (
    classify_anemia_dataset,
    classify_kidney_function,
    compute_apri,
)

def test_classify_anemia_dataset_basic():
    df = pd.DataFrame({
        'Hbn': [14, 12, 10, 7, 6, np.nan],
        'sex': ['m', 'm', 'f', 'f', 'f', 'm'],
        'diagnosis': ['I25.19', 'C34.1', 'Z00.0', 'E11.9', 'I25.19', 'E11.9']
    })
    result = classify_anemia_dataset(df)
    expected = [
        'Normal',           # male, 14
        'Mild Anemia',      # male, 12
        'Moderate Anemia',  # female, 10
        'Moderate Anemia',  # female, 7
        'Severe Anemia',    # female, 6
        'Unknown',          # NaN
    ]
    assert result['anemia_level'].tolist() == expected

def test_classify_anemia_dataset_pregnant():
    df = pd.DataFrame({
        'Hbn': [11, 10, 9, 6],
        'sex': ['f', 'f', 'f', 'f'],
        'diagnosis': ['O10', 'O20', 'O21', 'O99']
    })
    result = classify_anemia_dataset(df)
    expected = [
        'Normal',         # pregnant, 11
        'Mild Anemia',    # pregnant, 10
        'Moderate Anemia',# pregnant, 9
        'Severe Anemia',  # pregnant, 6
    ]
    assert result['anemia_level'].tolist() == expected

def test_classify_anemia_dataset_missing_columns():
    df = pd.DataFrame({'foo': [1, 2]})
    result = classify_anemia_dataset(df)
    assert 'anemia_level' in result.columns
    assert result['anemia_level'].isna().all()

def test_classify_kidney_function_basic():
    df = pd.DataFrame({'EPIGFR': [95, 75, 45, 15, np.nan]})
    result = classify_kidney_function(df)
    expected = ['normal', 'mild', 'moderate', 'severe', 'unknown']
    assert result['kidney_function'].tolist() == expected

def test_classify_kidney_function_missing_column():
    df = pd.DataFrame({'foo': [1, 2]})
    result = classify_kidney_function(df)
    assert (result['kidney_function'] == 'unknown').all()

def test_compute_apri_basic():
    df = pd.DataFrame({
        'ASAT': [20, 40, 80, 60],
        'THZn': [200, 100, 50, 30]
    })
    result = compute_apri(df)
    # APRI = ((ASAT/40)/THZn)*100
    apri_expected = [
        ((20/40)/200)*100,  # 0.25
        ((40/40)/100)*100,  # 1.0
        ((80/40)/50)*100,   # 4.0
        ((60/40)/30)*100,   # 5.0
    ]
    np.testing.assert_allclose(result['APRI'], apri_expected)
    expected_risk = [
        'no_fibrosis',      # 0.25
        'moderate_risk',    # 1.0
        'high_risk',        # 4.0
        'high_risk',        # 5.0
    ]
    assert result['liver_fibrosis_risk'].tolist() == expected_risk

def test_compute_apri_missing_columns():
    df = pd.DataFrame({'foo': [1, 2]})
    result = compute_apri(df)
    assert result['APRI'].isna().all()
    assert result['liver_fibrosis_risk'].isna().all()

def test_compute_apri_custom_columns():
    df = pd.DataFrame({
        'AST': [40],
        'PLT': [100]
    })
    result = compute_apri(df, asat_col='AST', platelet_col='PLT', ast_uln=40)
    assert np.isclose(result['APRI'].iloc[0], 1.0)
    assert result['liver_fibrosis_risk'].iloc[0] == 'moderate_risk'