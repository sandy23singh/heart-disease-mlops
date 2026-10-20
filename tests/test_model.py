import sys
from pathlib import Path

import joblib
import pandas as pd


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


def test_model_file_exists():
    """
    Check if trained model exists.
    """

    assert MODEL_PATH.exists(), (
        f"Model file not found: {MODEL_PATH}"
    )


def test_model_loads():
    """
    Check if model loads successfully.
    """

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_prediction_output():
    """
    Test prediction output.
    """

    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame({
        "age": [52],
        "sex": [1],
        "cp": [0],
        "trestbps": [130],
        "chol": [240],
        "fbs": [0],
        "restecg": [1],
        "thalach": [160],
        "exang": [0],
        "oldpeak": [1.5],
        "slope": [2],
        "ca": [0],
        "thal": [2]
    })

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]


def test_prediction_probability():
    """
    Check probability range.
    """

    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame({
        "age": [67],
        "sex": [1],
        "cp": [4],
        "trestbps": [160],
        "chol": [286],
        "fbs": [0],
        "restecg": [2],
        "thalach": [108],
        "exang": [1],
        "oldpeak": [2.5],
        "slope": [2],
        "ca": [3],
        "thal": [3]
    })

    probability = model.predict_proba(sample)

    confidence = probability.max()

    assert 0.0 <= confidence <= 1.0


def test_input_schema():
    """
    Validate expected feature count.
    """

    sample = pd.DataFrame({
        "age": [52],
        "sex": [1],
        "cp": [0],
        "trestbps": [130],
        "chol": [240],
        "fbs": [0],
        "restecg": [1],
        "thalach": [160],
        "exang": [0],
        "oldpeak": [1.5],
        "slope": [2],
        "ca": [0],
        "thal": [2]
    })

    assert sample.shape[1] == 13