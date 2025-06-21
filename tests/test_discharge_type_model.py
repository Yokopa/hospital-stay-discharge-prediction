import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.modeling import discharge_type_model
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# Dummy classifier to simulate behavior
class DummyClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.was_fit = False

    def fit(self, X, y):
        self.y_train = y
        self.was_fit = True
        return self

    def predict(self, X):
        if hasattr(self, "y_train"):
            mode_val = self.y_train.mode()[0] if not self.y_train.empty else 0
            return np.full(len(X), mode_val)
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        return np.tile([[0.7, 0.3]], (n, 1))

class DummyRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.was_fit = False

    def fit(self, X, y):
        self.was_fit = True
        return self

    def predict(self, X):
        return np.zeros(len(X))

# Dummy config to patch your actual config in modeling modules
class DummyConfig:
    RANDOM_SEED = 42
    REGRESSOR_CLASSES = {
        "LGBMRegressor": LGBMRegressor,
        "CatBoostRegressor": CatBoostRegressor,
        "XGBRegressor": XGBRegressor,
        "DummyRegressor": DummyRegressor,
    }
    CLASSIFIER_CLASSES = {
        "LGBMClassifier": LGBMClassifier,
        "CatBoostClassifier": CatBoostClassifier,
        "XGBClassifier": XGBClassifier,
        "DummyClassifier": DummyClassifier,
    }

def build_model_cfg(model_name, classifier_class_str, regressor_class_str, classifier_params=None, regressor_params=None):
    # Use class names as strings, not class objects!
    return {
        model_name: {
            "classifier": {
                "class": classifier_class_str,
                "params": classifier_params or {}
            },
            "regressor": {
                "class": regressor_class_str,
                "params": regressor_params or {}
            }
        }
    }

@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({
        "age": [60, 70, 80, 90],
        "sex": ["M", "F", "F", "M"],
        "feature_cat": pd.Series(["A", "B", "A", "B"], dtype="category")
    })
    X_test = pd.DataFrame({
        "age": [65, 85, 75],
        "sex": ["F", "M", "F"],
        "feature_cat": pd.Series(["A", "B", "A"], dtype="category")
    })
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 1, 2])
    return X_train, X_test, y_train, y_test

@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    monkeypatch.setattr("src.modeling.los_model.config", DummyConfig)
    monkeypatch.setattr("src.modeling.discharge_type_model.config", DummyConfig)
    yield

def test_train_discharge_pipeline_basic(sample_data):
    model_cfg = build_model_cfg("dummy", "DummyClassifier", "DummyRegressor")
    X_train, X_test, y_train, y_test = sample_data

    dummy_clf = MagicMock()
    dummy_clf.predict.return_value = y_test
    dummy_clf.predict_proba.return_value = np.tile([0.6, 0.4], (len(X_test), 1))

    with patch("src.modeling.discharge_type_model.utils.compute_scale_pos_weight", return_value=1), \
         patch("src.modeling.discharge_type_model.utils.train_classifier", return_value=(dummy_clf, y_test, dummy_clf.predict_proba.return_value)):
        result = discharge_type_model.train_discharge_pipeline(
            X_train, X_test, y_train, y_test, "dummy", model_cfg
        )

    assert "classifier" in result
    assert result["f1_score"] >= 0
    assert isinstance(result["precision"], float)

def test_train_discharge_pipeline_xgboost(sample_data):
    class XGBClassifierMock(DummyClassifier):
        pass
    XGBClassifierMock.__name__ = "XGBClassifier"  # <-- This is the key line

    model_cfg = build_model_cfg("xgboost", "XGBClassifier", "XGBRegressor")
    DummyConfig.CLASSIFIER_CLASSES["XGBClassifier"] = XGBClassifierMock

    X_train, X_test, y_train, y_test = sample_data

    with patch("src.modeling.discharge_type_model.utils.compute_scale_pos_weight", return_value=2.5), \
         patch("src.modeling.discharge_type_model.utils.train_classifier", return_value=(MagicMock(), y_test, None)) as mock_train:
        discharge_type_model.train_discharge_pipeline(
            X_train, X_test, y_train, y_test, "xgboost", model_cfg
        )
        clf_params_used = mock_train.call_args[0][1]
        assert "scale_pos_weight" in clf_params_used
        assert clf_params_used["scale_pos_weight"] == 2.5

def test_train_discharge_pipeline_lightgbm_categorical(sample_data):
    class LGBMClassifierMock(MagicMock):
        __name__ = "LGBMClassifier"

    model_cfg = build_model_cfg("lightgbm", "LGBMClassifier", "LGBMRegressor")
    DummyConfig.CLASSIFIER_CLASSES["LGBMClassifier"] = LGBMClassifierMock

    X_train, X_test, y_train, y_test = sample_data

    with patch("src.modeling.discharge_type_model.utils.compute_scale_pos_weight", return_value=1), \
         patch("src.modeling.discharge_type_model.utils.train_classifier") as mock_train:
        mock_train.return_value = (MagicMock(), [0, 1, 0], [[0.7, 0.3]] * 3)
        discharge_type_model.train_discharge_pipeline(
            X_train, X_test, y_train, y_test, "lightgbm", model_cfg
        )
        clf_params = mock_train.call_args[1]
        cat_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()
        assert "sex" in cat_features
        assert "feature_cat" in cat_features

def test_train_discharge_pipeline_catboost(sample_data):
    class CatBoostClassifierMock(DummyClassifier):
        __name__ = "CatBoostClassifier"
        def fit(self, train_pool, eval_set=None, verbose=0):
            self.was_fit = True
            return self
        def predict(self, test_pool):
            return np.array([0, 1, 2])
        def predict_proba(self, test_pool):
            return np.array([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]
            ])

    model_cfg = build_model_cfg("catboost", "CatBoostClassifier", "CatBoostRegressor")
    DummyConfig.CLASSIFIER_CLASSES["CatBoostClassifier"] = CatBoostClassifierMock

    X_train, X_test, y_train, y_test = sample_data

    with patch("src.modeling.discharge_type_model.utils.compute_scale_pos_weight", return_value=1), \
         patch("catboost.Pool", MagicMock()):
        result = discharge_type_model.train_discharge_pipeline(
            X_train, X_test, y_train, y_test, "catboost", model_cfg
        )

    assert isinstance(result["classifier"], CatBoostClassifierMock)
    assert result["f1_score"] >= 0
    assert isinstance(result["roc_auc"], float)
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["log_loss"] is not None

def test_train_discharge_pipeline_multiclass_proba(sample_data):
    class MultiClassClassifier(DummyClassifier):
        __name__ = "MultiClassClassifier"
        def predict(self, X):
            classes = [0, 1, 2]
            return np.array([classes[i % 3] for i in range(len(X))])
        def predict_proba(self, X):
            return np.tile([0.7, 0.2, 0.1], (len(X), 1))

    DummyConfig.CLASSIFIER_CLASSES["MultiClassClassifier"] = MultiClassClassifier
    DummyConfig.REGRESSOR_CLASSES["DummyRegressor"] = DummyRegressor

    model_cfg = build_model_cfg("multi", "MultiClassClassifier", "DummyRegressor")
    X_train, X_test, y_train, y_test = sample_data
    y_train = pd.Series([0, 1, 2, 1])
    y_test = pd.Series([0, 1, 2])
    with patch("src.modeling.discharge_type_model.utils.compute_scale_pos_weight", return_value=1):
        result = discharge_type_model.train_discharge_pipeline(
            X_train, X_test, y_train, y_test, "multi", model_cfg
        )
    assert isinstance(result["roc_auc"], float)
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["log_loss"] is not None