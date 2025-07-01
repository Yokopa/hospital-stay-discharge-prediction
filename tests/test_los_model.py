import pytest
import numpy as np
import pandas as pd
from src.modeling import los_model
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

class DummyConfig:
    RANDOM_SEED = 42
    REGRESSOR_CLASSES = {
        "LGBMRegressor": LGBMRegressor,
        "CatBoostRegressor": CatBoostRegressor,
        "XGBRegressor": XGBRegressor,
    }
    CLASSIFIER_CLASSES = {
        "LGBMClassifier": LGBMClassifier,
        "CatBoostClassifier": CatBoostClassifier,
        "XGBClassifier": XGBClassifier
    } 

@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    monkeypatch.setattr(los_model, "config", DummyConfig)

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
    X_test = pd.DataFrame({"a": [5, 6, 7], "b": [6, 7, 8]})  # Now 3 rows
    y_train = np.array([2, 3, 4, 5])
    y_test = np.array([0, 1, 2])  # Also 3 values
    return X_train, X_test, y_train, y_test

def test_train_los_regression_baseline_lgbm(sample_data):
    model_cfg_lgbm = {
        "regressor": {
            "class": "LGBMRegressor",
            "params": {"n_estimators": 5, "random_state": 42}
        }
    }
    X_train, X_test, y_train, y_test = sample_data
    result = los_model.train_los_regression_baseline(
        X_train, X_test, y_train, y_test, model_cfg_lgbm
    )
    assert "regressor" in result
    assert "rmse" in result
    assert "mae" in result
    assert "r2" in result
    assert isinstance(result["rmse"], float)
    assert isinstance(result["mae"], float)
    assert isinstance(result["r2"], float)
    assert hasattr(result["regressor"], "predict")

def test_train_los_multiclass_baseline_lgbm(sample_data):
    model_cfg_lgbm = {
        "classifier": {
            "class": "LGBMClassifier",
            "params": {"n_estimators": 5, "random_state": 42}
        }
    }
    X_train, X_test, _, _ = sample_data
    y_train = np.array([0, 1, 2, 1])
    y_test = np.array([0, 1, 2])
    bins = [0, 1, 2, 3]
    result = los_model.train_los_multiclass_baseline(
        X_train, X_test, y_train, y_test, bins, model_cfg_lgbm
    )
    assert "classifier" in result
    assert "f1_score" in result
    assert "precision" in result
    assert "recall" in result
    assert "balanced_accuracy" in result
    assert isinstance(result["f1_score"], float)
    assert isinstance(result["precision"], float)
    assert isinstance(result["recall"], float)
    assert isinstance(result["balanced_accuracy"], float)

def test_train_los_pipeline_lgbm(sample_data):
    model_cfg_lgbm = {
        "lgbm": {
            "classifier": {
                "class": "LGBMClassifier",
                "params": {"n_estimators": 5, "random_state": 42}
            },
            "regressor": {
                "class": "LGBMRegressor",
                "params": {"n_estimators": 5, "random_state": 42}
            }
        }
    }
    X_train, X_test, _, _ = sample_data
    y_train = np.array([1, 2, 3, 4])
    y_test = np.array([1, 2, 3])
    result = los_model.train_los_two_step_pipeline(
        X_train, X_test, y_train, y_test, threshold=2, model_name="lgbm", model_cfg=model_cfg_lgbm
    )
    assert "classifier" in result
    assert "regressor" in result
    assert "f1_score" in result
    assert "precision" in result
    assert "recall" in result
    assert "balanced_accuracy" in result
    assert isinstance(result["f1_score"], float)
    assert isinstance(result["precision"], float)
    assert isinstance(result["recall"], float)
    assert isinstance(result["balanced_accuracy"], float)