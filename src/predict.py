import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"

class HeartDiseasePredictor:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, patient_data):

        df = pd.DataFrame([patient_data])

        prediction = self.model.predict(df)[0]

        probability = self.model.predict_proba(df)[0]

        confidence = max(probability)

        return {
            "prediction": int(prediction),
            "confidence": float(confidence)
        }


if __name__ == "__main__":

    sample_patient = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }

    predictor = HeartDiseasePredictor()

    result = predictor.predict(
        sample_patient
    )

    print(result)

    if result["prediction"] == 1:
        print(
            f"Heart Disease Detected "
            f"(Confidence: {result['confidence']:.2%})"
        )
    else:
        print(
            f"No Heart Disease Detected "
            f"(Confidence: {result['confidence']:.2%})"
        )