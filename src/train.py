import os
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_data():
    """
    Load Heart Disease Dataset
    """

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    df = pd.concat([X, y], axis=1)

    df["target"] = (df["num"] > 0).astype(int)

    X = df.drop(
        columns=["num", "target"]
    )

    y = df["target"]

    return X, y


def build_preprocessor(features):

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


def evaluate_model(
    model,
    X,
    y,
    X_test,
    y_test
):

    preds = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds
    )

    recall = recall_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds
    )

    roc = roc_auc_score(
        y_test,
        preds
    )

    cv_score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    ).mean()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc,
        "cv_accuracy": cv_score
    }


def train_logistic(
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    X,
    y
):

    with mlflow.start_run(
        run_name="LogisticRegression"
    ):

        model = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X,
            y,
            X_test,
            y_test
        )

        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        for metric, value in metrics.items():
            mlflow.log_metric(
                metric,
                value
            )

        mlflow.sklearn.log_model(
            model,
            name="logistic_model",
            serialization_format="cloudpickle"
        )

        print("\nLogistic Regression")
        print(metrics)

        return model, metrics["accuracy"]


def train_random_forest(
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
    X,
    y
):

    with mlflow.start_run(
        run_name="RandomForest"
    ):

        model = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X,
            y,
            X_test,
            y_test
        )

        mlflow.log_param(
            "model",
            "RandomForest"
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "max_depth",
            5
        )

        for metric, value in metrics.items():
            mlflow.log_metric(
                metric,
                value
            )

        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            serialization_format="cloudpickle"
        )

        print("\nRandom Forest")
        print(metrics)

        return model, metrics["accuracy"]


def main():

    os.makedirs(
        "../models",
        exist_ok=True
    )

    mlflow.set_experiment(
        "Heart Disease Prediction"
    )

    X, y = load_data()

    preprocessor = build_preprocessor(
        X.columns
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    logistic_model, logistic_acc = train_logistic(
        preprocessor,
        X_train,
        y_train,
        X_test,
        y_test,
        X,
        y
    )

    rf_model, rf_acc = train_random_forest(
        preprocessor,
        X_train,
        y_train,
        X_test,
        y_test,
        X,
        y
    )

    if rf_acc > logistic_acc:

        best_model = rf_model
        best_name = "Random Forest"

    else:

        best_model = logistic_model
        best_name = "Logistic Regression"

    joblib.dump(
        best_model,
        "../models/model.pkl"
    )

    print(
        f"\nBest Model Saved: {best_name}"
    )


if __name__ == "__main__":
    main()