# 🏎️ F1 Safety Car Detector (V1 MVP)

A real-time Formula 1 Safety Car detection service built as an end-to-end MLOps pipeline. Given a driver's current track position and lap time, the API estimates the probability that a Safety Car is currently deployed on track.

> **Scope Note:** This is a **real-time incident detector**, not a pre-race crystal ball. Because crashes and mechanical failures happen unpredictably, the model evaluates live pace shifts rather than forecasting which future lap an accident will occur on.

---

## 🛠️ What This Pipeline Does

1. **Automated Data Ingestion:** Pulls historical lap-by-lap race telemetry from the Ergast F1 API.
2. **Leak-Free Transformation:** Converts lap strings into total seconds, handles missing values with `SimpleImputer`, and scales features using `StandardScaler`.
3. **Imbalanced Classification:** Trains an `XGBClassifier` tuned with `scale_pos_weight` to handle the rarity of Safety Car laps compared to green-flag racing, tracked via MLflow.
4. **Low-Latency Serving:** Serves predictions through a FastAPI REST endpoint (`/predict`) using Pydantic input validation and startup `lifespan` memory caching.

---

## 🚀 Quickstart

### 1. Start the API Server
```bash
uvicorn app:app --reload
```

### 2. Query the Prediction Endpoint
Send a `POST` request to `/predict` with a car's current position and lap time in seconds (for example, `1:53.99` converts to `113.99` seconds):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"position": 15, "lap_time_sec": 113.99}'
```

**Sample Response:**
```json
{
  "is_safety_car": 0,
  "safety_car_probability": 0.2138
}
```

---

## 🔍 Engineering Lessons & Known Flaws (V1)

* **Catching Target Leakage:** Early iterations included race metadata (`season`, `round`, `lap_number`), producing a suspicious `1.0000` ROC-AUC because the model memorized specific historical races. Stripping those identifiers dropped ROC-AUC to a realistic **`0.8551`** (Log Loss: `0.4346`), forcing the model to rely purely on on-track pace.
* **The Single-Driver Limitation:** V1 evaluates one driver's lap at a time. While a Safety Car slows the entire field down, a single driver making a routine pit stop (~20–25s loss) or running on wet tires in heavy rain also records slow lap times—which can trigger false positives without grid-wide context.

---

## 🗺️ V2 Roadmap

* **Grid-Wide Aggregation:** Compare a driver's lap time against the median pace of all 20 cars on that lap to separate isolated pit stops from full-course neutralizations.
* **Environmental Context:** Integrate track type (street vs. traditional) and live weather conditions.
* **Production Ops:** Containerize the FastAPI service with Docker and add automated CI/CD testing.