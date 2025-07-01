import pytest
import pandas as pd
import numpy as np
import sys
import os
from src.data_preparation import integrate_data

# Ensure the module can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Mock config and utils for isolated testing
class DummyConfig:
    MISSING_THRESHOLD = 50.0
    REMOVE_TESTS = ['REMOVE_ME']
    CLEANED_CLINICAL_PATH = "dummy_clinical.csv"
    CLEANED_LAB_PATH = "dummy_lab.csv"
    REFERENCE_TABLE_PATH = "dummy_reference.csv"
    MERGED_DATA_PATH = "dummy_merged.csv"
    CLEANED_MERGED_DATA_PATH = "dummy_cleaned_merged.csv"
    LAB_TEST_STATISTICS = "dummy_stats.csv"

class DummyUtils:
    @staticmethod
    def load_csv(path):
        if "clinical" in path:
            return pd.DataFrame({
                "patient_id": [1, 2],
                "case_id": [10, 20],
                "age": [25, 17],
                "other": ["a", "b"]
            })
        elif "lab" in path:
            return pd.DataFrame({
                "patient_id": [1, 2, 1],
                "case_id": [10, 20, 10],
                "test_abbr": ["A", "B", "REMOVE_ME"],
                "test_name": ["TestA", "TestB", "RemoveTest"],
                "method_number": [1, 1, 1],
                "unit": ["mg", "mg", "mg"],
                "numeric_result": [5.0, -1.0, 7.0]
            })
        elif "merged" in path:
            return pd.DataFrame({
                "INRiH": [9999.0, 2.0],
                "QUHD": [9999.0, 1.0],
                "Hkn": [42.00, 0],
                "Hbn": [0, 2.0],
                "THZn": [0, 2.0],
                "tHb": [0, 2.0],
                "ALAT": [0, 2.0],
                "ASAT": [0, 2.0],
                "CR": [0, 2.0],
                "BIg": [0, 2.0],
                "CRP": [0, 2.0],
                "GGT": [0, 2.0],
                "GL": [0, 2.0],
                "UREA": [0, 2.0],
                "age": [25, 17]
            })
        else:
            return pd.DataFrame()

    @staticmethod
    def save_csv(df, path):
        # Do nothing for testing
        pass

    @staticmethod
    def remove_duplicates(df):
        return df.drop_duplicates()

    @staticmethod
    def find_negative_columns(df, start_col=8):
        # Return columns with negative values
        cols = df.columns[start_col:]
        return [col for col in cols if (df[col] < 0).any()]

    @staticmethod
    def replace_negatives_with_nan(df, cols):
        for col in cols:
            df[col] = df[col].mask(df[col] < 0, np.nan)
        return df

    @staticmethod
    def configure_logging(verbose=True):
        pass

    @staticmethod
    def generate_summary_statistics(df, start_col=4, save_path=None, verbose=False):
        pass

@pytest.fixture(autouse=True)
def patch_config_and_utils(monkeypatch):
    monkeypatch.setattr(integrate_data, "config", DummyConfig)
    monkeypatch.setattr(integrate_data, "utils", DummyUtils)

def test_integrate_data_basic():
    clinical = pd.DataFrame({
        "patient_id": [1, 2],
        "case_id": [10, 20],
        "age": [25, 17],
        "other": ["a", "b"]
    })
    lab = pd.DataFrame({
        "patient_id": [1, 2, 1],
        "case_id": [10, 20, 10],
        "test_abbr": ["A", "B", "REMOVE_ME"],
        "test_name": ["TestA", "TestB", "RemoveTest"],
        "method_number": [1, 1, 1],
        "unit": ["mg", "mg", "mg"],
        "numeric_result": [5.0, -1.0, 7.0]
    })
    result = integrate_data.integrate_data(
        clinical_data=clinical,
        lab_data=lab,
        missing_threshold=50.0,
        remove_tests=['REMOVE_ME'],
        save=False,
        verbose=True
    )
    print(result)
    print(result.columns)
    # Only test_abbr A and B should remain, REMOVE_ME is filtered
    assert "REMOVE_ME" not in result.columns
    # # Negative value in B should be replaced with NaN
    # assert np.isnan(result.loc[result["patient_id"] == 2, "B"]).all() # modify function to be more precise and flexible
    # Should merge on patient_id and case_id
    assert set(result["patient_id"]) == {1, 2}

def test_replace_invalid_values():
    df = DummyUtils.load_csv("dummy_merged.csv")
    cleaned = integrate_data.replace_invalid_values(df_in=df, save=False, verbose=True)
    # INRiH and QUHD 9999.0 replaced with NaN
    assert np.isnan(cleaned.loc[0, "INRiH"])
    assert np.isnan(cleaned.loc[0, "QUHD"])
    # Hkn 42.00 replaced with 0.42, 0 replaced with NaN
    assert cleaned.loc[0, "Hkn"] == 0.42
    assert np.isnan(cleaned.loc[1, "Hkn"])
    # Hbn, THZn, tHb: 0 replaced with NaN
    assert np.isnan(cleaned.loc[0, "Hbn"])
    assert np.isnan(cleaned.loc[0, "THZn"])
    assert np.isnan(cleaned.loc[0, "tHb"])
    # ALAT, ASAT, CR: 0 replaced with 2.5
    assert cleaned.loc[0, "ALAT"] == 2.5
    assert cleaned.loc[0, "ASAT"] == 2.5
    assert cleaned.loc[0, "CR"] == 2.5
    # BIg, CRP, GGT: 0 replaced with 1.5
    assert cleaned.loc[0, "BIg"] == 1.5
    assert cleaned.loc[0, "CRP"] == 1.5
    assert cleaned.loc[0, "GGT"] == 1.5
    # GL: 0 replaced with 0.10
    assert cleaned.loc[0, "GL"] == 0.10
    # UREA: 0 replaced with 0.25
    assert cleaned.loc[0, "UREA"] == 0.25

def test_filter_adults():
    df = pd.DataFrame({
        "age": [17, 18, 25, 130],
        "other": [1, 2, 3, 4]
    })
    filtered = integrate_data.filter_adults(df, age_column="age", min_age=18, max_age=120, save=False)
    # Only ages 18 and 25 should remain
    assert set(filtered["age"]) == {18, 25}
    assert len(filtered) == 2