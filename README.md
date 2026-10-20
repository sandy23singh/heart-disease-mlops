# Heart Disease Prediction MLOps Project

## Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting the risk of heart disease using the UCI Heart Disease Dataset.

The solution covers:

- Data Acquisition
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Training
- Experiment Tracking with MLflow
- Model Packaging
- FastAPI Inference API
- Docker Containerization
- CI/CD using GitHub Actions
- Kubernetes Deployment
- Automated Testing

---

## Problem Statement

Develop a machine learning classifier capable of predicting whether a patient is at risk of heart disease based on clinical and demographic health information.

---

## Dataset

Source:

UCI Machine Learning Repository

Dataset:

Heart Disease Dataset

Features:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

Target:

- 0 = No Heart Disease
- 1 = Heart Disease Present

---

## Project Structure

```text
heart-disease-mlops/
│
├── api/
│   └── app.py
│
├── deployment/
│   ├── deployment.yaml
│   └── deploymentservice.yaml
│
├── models/
│   └── model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Inference.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── tests/
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy

### Experiment Tracking

- MLflow

### API

- FastAPI
- Uvicorn

### Testing

- Pytest

### Containerization

- Docker

### Deployment

- Kubernetes
- Minikube

### CI/CD

- GitHub Actions

---

## Setup Instructions

### Clone Repository

```bash
git clone <repository-url>

cd heart-disease-mlops
```

### Create Environment

```bash
conda create -n heartmlops python=3.11 -y

conda activate heartmlops
```

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## Model Training

Run:

```bash
python src/train.py
```

Artifacts generated:

```text
models/model.pkl
```

MLflow logs:

```text
mlruns/
```

---

## Run Unit Tests

Execute:

```bash
python -m pytest tests -v
```

Expected:

```text
10 passed
```

---

## Inference

Execute:

```bash
python src/predict.py
```

Example Output:

```json
{
  "prediction": 0,
  "confidence": 0.91
}
```

---

## Run FastAPI Application

Start server:

```bash
uvicorn api.app:app --reload
```

Application URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Example

### POST /predict

Request:

```json
{
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
```

Response:

```json
{
  "prediction": 0,
  "risk": "No Heart Disease Risk",
  "confidence": 0.91
}
```

---

## Docker

### Build Image

```bash
docker build -t heart-disease-mlops .
```

### Run Container

```bash
docker run -p 8000:8000 heart-disease-mlops
```

### Verify

```text
http://localhost:8000/docs
```

---

## Kubernetes Deployment

Apply Deployment:

```bash
kubectl apply -f deployment/deployment.yaml
```

Apply Service:

```bash
kubectl apply -f deployment/deploymentservice.yaml
```

Check Pods:

```bash
kubectl get pods
```

Check Services:

```bash
kubectl get svc
```

For Minikube:

```bash
minikube service heart-disease-mlops-service
```

---

## CI/CD Pipeline

Implemented using GitHub Actions.

Pipeline Stages:

1. Dependency Installation
2. Model Training
3. Unit Testing
4. MLflow Validation
5. Docker Build
6. Artifact Upload
7. Code Linting

Workflow Location:

```text
.github/workflows/ci-cd.yml
```

---

## Experiment Tracking

MLflow tracks:

- Model Parameters
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross Validation Accuracy

Run MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

## Test Results

| Test | Status |
|--------|--------|
| Model Loading | Passed |
| Prediction Validation | Passed |
| Probability Validation | Passed |
| Input Schema Validation | Passed |
| Missing Value Handling | Passed |
| Preprocessing Validation | Passed |

---

## Future Enhancements

- XGBoost Model
- Hyperparameter Tuning
- Prometheus Monitoring
- Grafana Dashboard
- Helm Charts
- Cloud Deployment (AKS/EKS/GKE)

---

## Author

Sandeep Singh

Machine Learning Operations (MLOps)

Heart Disease Prediction Project