# 🏎️ F1 Safety Car Predictor

[![Continuous Integration](https://github.com/Oshobugie/f1_safety_car_predictorr/actions/workflows/ci.yml/badge.svg)](https://github.com/Oshobugie/f1_safety_car_predictorr/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-EB5E28)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)

An end-to-end MLOps machine learning pipeline and containerized FastAPI microservice that predicts Formula 1 Safety Car deployments (`is_safety_car`) and deployment probabilities from real-time lap telemetry (`position` and `lap_time_sec`).

---

## 🌐 Live API & Interactive Documentation

Hosted as a containerized Docker web service on **Render** with automated deployments gated behind GitHub Actions CI checks:

* **Interactive Swagger UI (`/docs`):** [https://f1-safety-car-predictor.onrender.com/docs](https://f1-safety-car-predictor.onrender.com/docs)
* **Health Check Endpoint (`/`):** [https://f1-safety-car-predictor.onrender.com/](https://f1-safety-car-predictor.onrender.com/)

> **Note:** This service runs on Render's free tier and spins down after 15 minutes of inactivity. The first request after an idle period may take **30–50 seconds** to wake the container.

### Sample Live Prediction Request

```bash
curl -X POST "[https://f1-safety-car-predictor.onrender.com/predict](https://f1-safety-car-predictor.onrender.com/predict)" \
  -H "Content-Type: application/json" \
  -d '{"position": 15, "lap_time_sec": 113.99}'
```

**Response:**
```json
{
  "is_safety_car": 1,
  "safety_car_probability": 0.8742
}
```

---

## 🏗️ System Architecture & CI/CD Pipeline

```text
Ergast F1 API (2023 Season)
        │
        ▼
[1. Data Ingestion] ──► races_raw.json & races_ingested.csv
        │
        ▼
[2. Data Transformation] ──► Median Imputation + StandardScaler ──► preprocessor.joblib
        │
        ▼
[3. Model Trainer] ──► Stratified Split + XGBoost + MLflow ──► model.joblib
        │
        ▼
[4. FastAPI Service] ──► Packaged in Docker (Port 7860)
        │
        ▼
[5. GitHub Actions CI] ──► Runs pytest test_app.py on push to main
        │
        ▼ (Only after CI passes)
[6. Render Cloud CD] ──► Auto-builds & deploys live container
```

### Key Engineering Highlights
* **Leakage-Free Feature Engineering:** Converts lap timings (`M:SS.mmm`) to numeric seconds, maps ground-truth 2023 Safety Car laps, and drops race identifiers (`round`, `lap_number`, `driverId`, `raceName`) prior to scaling so the model generalizes strictly on telemetry.
* **Class Imbalance Handling:** Applies stratified train-test splitting (`stratify=y`) and computes `scale_pos_weight` dynamically from the training split to account for rare Safety Car laps.
* **Experiment Tracking:** Logs hyperparameters, `ROC-AUC`, `Log Loss`, `Accuracy`, and serialized models to MLflow (`F1_Safety_Car_Prediction`).
* **Automated Quality Gate:** Every push to `main` triggers `.github/workflows/ci.yml` to run `pytest test_app.py`, verifying artifact loading and `/predict` response schemas before Render deploys a new container.

---

## 📂 Project Structure

```text
f1_safety_car_predictorr/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI test pipeline
├── artifacts/
│   ├── data_transformation/
│   │   └── preprocessor.joblib     # Fitted imputer + StandardScaler pipeline
│   └── model_trainer/
│       └── model.joblib            # Trained XGBoost classifier
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Ergast API extraction & JSON normalization
│   │   ├── data_transformation.py  # Feature engineering, labeling & preprocessing
│   │   └── model_trainer.py        # Stratified XGBoost training & MLflow logging
│   ├── config/
│   │   └── configuration.py        # YAML-to-dataclass configuration manager
│   ├── entity/                     # Config & artifact dataclasses
│   ├── exception.py                # Custom exception wrapper with file/line trace
│   └── logger.py                   # Structured timestamped logging
├── app.py                          # FastAPI inference service
├── main.py                         # End-to-end training pipeline entrypoint
├── test_app.py                     # Automated FastAPI smoke tests
├── Dockerfile                      # Slim Python 3.11 production container
├── params.yaml                     # Pipeline paths and configuration parameters
└── requirements.txt                # Python dependencies
```

---

## 🚀 Running Locally

### 1. Local Python Environment
```bash
# Clone the repository
git clone [https://github.com/Oshobugie/f1_safety_car_predictorr.git](https://github.com/Oshobugie/f1_safety_car_predictorr.git)
cd f1_safety_car_predictorr

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Re-run the full training pipeline
python main.py

# Run automated API tests
pytest test_app.py

# Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### 2. Docker Container
```bash
docker build -t f1-safety-car-predictor .
docker run -p 7860:7860 f1-safety-car-predictor
```
Then open `http://localhost:7860/docs` in your browser.