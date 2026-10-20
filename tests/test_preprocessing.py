import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def create_preprocessor(features):

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                features
            )
        ]
    )


def test_preprocessor_creation():
    """
    Verify preprocessing object is created.
    """

    features = [
        "age",
        "chol",
        "thalach"
    ]

    preprocessor = create_preprocessor(
        features
    )

    assert preprocessor is not None


def test_missing_value_imputation():
    """
    Verify missing values are handled.
    """

    df = pd.DataFrame({
        "age": [63, np.nan, 41],
        "chol": [233, 286, np.nan],
        "thalach": [150, 108, 172]
    })

    preprocessor = create_preprocessor(
        df.columns
    )

    transformed = preprocessor.fit_transform(
        df
    )

    assert np.isnan(transformed).sum() == 0


def test_output_shape():
    """
    Output column count should remain unchanged.
    """

    df = pd.DataFrame({
        "age": [63, 67, 41],
        "chol": [233, 286, 204],
        "thalach": [150, 108, 172]
    })

    preprocessor = create_preprocessor(
        df.columns
    )

    transformed = preprocessor.fit_transform(
        df
    )

    assert transformed.shape[0] == 3
    assert transformed.shape[1] == 3


def test_scaling_applied():
    """
    StandardScaler should center around zero.
    """

    df = pd.DataFrame({
        "age": [40, 50, 60],
        "chol": [200, 240, 280]
    })

    preprocessor = create_preprocessor(
        df.columns
    )

    transformed = preprocessor.fit_transform(
        df
    )

    mean_value = transformed.mean()

    assert abs(mean_value) < 1e-6


def test_feature_count():
    """
    Feature count should match input.
    """

    df = pd.DataFrame({
        "age": [63],
        "sex": [1],
        "cp": [1],
        "trestbps": [145],
        "chol": [233],
        "fbs": [1],
        "restecg": [2],
        "thalach": [150],
        "exang": [0],
        "oldpeak": [2.3],
        "slope": [3],
        "ca": [0],
        "thal": [6]
    })

    preprocessor = create_preprocessor(
        df.columns
    )

    transformed = preprocessor.fit_transform(
        df
    )

    assert transformed.shape[1] == 13
