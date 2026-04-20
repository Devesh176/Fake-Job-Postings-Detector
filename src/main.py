"""
Fake Job Posting Detector — FastAPI Backend
Handles prediction, feedback logging, inference storage, and monitoring.
"""

import os
import uuid
import json
import logging
import time
import pickle
from datetime import datetime
from typing import Optional

import scipy.sparse as sp
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import sqlite3

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_request_count_total",
    "Total API requests",
    ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
)
ERROR_COUNT = Counter(
    "api_error_count_total",
    "Total API errors"
)
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Model prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)
FRAUD_RATE = Gauge(
    "model_fraud_prediction_rate",
    "Current fraud prediction rate"
)

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake Job Posting Detector",
    description="Detects fraudulent job postings using ML",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "data/production/model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "data/production/tfidf_vectorizer.pkl")
BASELINE_PATH = os.getenv("BASELINE_PATH", "app/baselines/training_baseline.json")
DB_CONN_STR = os.getenv("DATABASE_URL", "sqlite:///data/inference.db")
MODEL_SERVE_MODE = os.getenv("MODEL_SERVE_MODE", "local")   # "local" or "mlflow"
MLFLOW_SERVE_URL = os.getenv("MLFLOW_SERVE_URL", "http://mlflow_model:5001/")

# ── Global State ──────────────────────────────────────────────────────────────
model = None
vectorizer = None
baselines = None


# ── Database Setup ────────────────────────────────────────────────────────────
def get_db():
    """Get a database connection."""
    return sqlite3.connect("data/inference.db")


def init_db():
    """Create inference_log table if it doesn't exist."""
    try:
        os.makedirs("data", exist_ok=True)
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inference_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                title TEXT,
                description TEXT,
                company_profile TEXT,
                requirements TEXT,
                employment_type TEXT,
                has_company_logo INTEGER,
                has_questions INTEGER,
                salary_range TEXT,
                prediction TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                inference_latency_ms REAL,
                user_label TEXT
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


# ── Model Loading ─────────────────────────────────────────────────────────────
def load_model():
    """Load model and vectorizer from production path."""
    global model, vectorizer, baselines
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info("Model and vectorizer loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        model = None
        vectorizer = None

    try:
        with open(BASELINE_PATH, "r") as f:
            baselines = json.load(f)
        logger.info("Baselines loaded successfully.")
    except FileNotFoundError:
        logger.warning("Baselines file not found — drift detection disabled.")
        baselines = None


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    load_model()


# ── Schemas ───────────────────────────────────────────────────────────────────
class JobPosting(BaseModel):
    title: str
    description: str
    company_profile: Optional[str] = ""
    requirements: Optional[str] = ""
    employment_type: Optional[str] = ""
    has_company_logo: Optional[int] = 0
    has_questions: Optional[int] = 0
    salary_range: Optional[str] = ""
    telecommuting: Optional[int] = 0
    location: Optional[str] = ""
    benefits: Optional[str] = ""
    user_label: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction_id: str
    prediction: str
    fraud_probability: float
    confidence: float
    inference_latency_ms: float
    risk_level: str
    key_signals: list


# ── Helper: Feature Engineering ───────────────────────────────────────────────
def build_features(posting: JobPosting):
    """Build feature vector from a job posting — EXACTLY matches preprocess.py."""
    import scipy.sparse as sp

    combined_text = " ".join([
        posting.title or "",
        posting.location or "",
        posting.description or "",
        posting.company_profile or "",
        posting.requirements or "",
        posting.benefits or "",
    ])

    has_salary = 1 if posting.salary_range else 0
    has_company_profile = 1 if (posting.company_profile and posting.company_profile.strip() != "") else 0
    has_requirements = 1 if (posting.requirements and posting.requirements.strip() != "") else 0
    has_benefits = 1 if (posting.benefits and posting.benefits.strip() != "") else 0

    X_text = vectorizer.transform([combined_text])

    X_meta = sp.csr_matrix([[
        int(posting.has_company_logo or 0),
        int(posting.has_questions or 0),
        has_salary,
        has_company_profile,
        has_requirements,
        int(posting.telecommuting or 0),
        has_benefits,
    ]])

    return sp.hstack([X_text, X_meta])


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    elif prob < 0.8:
        return "HIGH"
    return "CRITICAL"


def get_key_signals(posting: JobPosting, prob: float) -> list:
    """Return human-readable signals that contributed to the prediction."""
    signals = []
    if not posting.company_profile:
        signals.append("No company profile provided")
    if not posting.has_company_logo:
        signals.append("No company logo")
    if not posting.salary_range:
        signals.append("Salary range not disclosed")
    if not posting.requirements:
        signals.append("No requirements listed")
    if not posting.has_questions:
        signals.append("No screening questions")
    if len(posting.description) < 100:
        signals.append("Very short job description")
    if prob > 0.5 and not signals:
        signals.append("Suspicious text patterns detected")
    return signals[:4]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
def ready():
    if MODEL_SERVE_MODE == "local" and (model is None or vectorizer is None):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_loaded": True, "mode": MODEL_SERVE_MODE}


@app.get("/mode")
def get_mode():
    return {
        "mode": MODEL_SERVE_MODE,
        "mlflow_url": MLFLOW_SERVE_URL if MODEL_SERVE_MODE == "mlflow" else None
    }


def build_response(posting: JobPosting, fraud_prob: float, duration: float) -> PredictionResponse:
    prediction = "Fraudulent" if fraud_prob >= 0.5 else "Legitimate"
    confidence = fraud_prob if fraud_prob >= 0.5 else 1 - fraud_prob
    latency_ms = duration * 1000
    prediction_id = str(uuid.uuid4())

    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000)
    PREDICTION_CONFIDENCE.observe(confidence)
    FRAUD_RATE.set(fraud_prob)

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO inference_log (
                id, timestamp, title, description, company_profile,
                requirements, employment_type, has_company_logo,
                has_questions, salary_range, prediction,
                fraud_probability, confidence, inference_latency_ms,
                user_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id, datetime.utcnow().isoformat(),
            posting.title, posting.description, posting.company_profile,
            posting.requirements, posting.employment_type,
            posting.has_company_logo, posting.has_questions,
            posting.salary_range, prediction,
            fraud_prob, confidence, latency_ms,
            posting.user_label
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as db_err:
        logger.error(f"Failed to log to DB: {db_err}")

    logger.info(
        f"[{MODEL_SERVE_MODE}] Prediction: {prediction} | Prob: {fraud_prob:.3f} | "
        f"Latency: {latency_ms:.2f}ms | ID: {prediction_id}"
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        prediction=prediction,
        fraud_probability=round(fraud_prob, 4),
        confidence=round(confidence, 4),
        inference_latency_ms=round(latency_ms, 2),
        risk_level=get_risk_level(fraud_prob),
        key_signals=get_key_signals(posting, fraud_prob)
    )


def predict_local(posting: JobPosting) -> PredictionResponse:
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Local model not loaded. Run DVC pipeline first.")
    start = time.time()
    X = build_features(posting)
    fraud_prob = float(model.predict_proba(X)[0][1])
    return build_response(posting, fraud_prob, time.time() - start)


def predict_mlflow(posting: JobPosting) -> PredictionResponse:
    import requests as req
    start = time.time()
    try:
        payload = {"dataframe_records": [{
            "title": posting.title,
            "description": posting.description,
            "company_profile": posting.company_profile or "",
            "requirements": posting.requirements or "",
            "employment_type": posting.employment_type or "",
            "has_company_logo": posting.has_company_logo,
            "has_questions": posting.has_questions,
            "salary_range": posting.salary_range or "",
        }]}
        res = req.post(MLFLOW_SERVE_URL, json=payload, timeout=5)
        res.raise_for_status()
        fraud_prob = float(res.json()["predictions"][0])
        return build_response(posting, fraud_prob, time.time() - start)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow server error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict(posting: JobPosting, request: Request):
    REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
    try:
        if MODEL_SERVE_MODE == "mlflow":
            return predict_mlflow(posting)
        return predict_local(posting)
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def get_history(limit: int = 20):
    try:
        conn = get_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT id, timestamp, title, prediction, fraud_probability,
                   confidence, inference_latency_ms, user_label
            FROM inference_log
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"history": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Return aggregate prediction statistics."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as total_predictions,
                SUM(CASE WHEN prediction = 'Fraudulent' THEN 1 ELSE 0 END) as fraud_count,
                SUM(CASE WHEN prediction = 'Legitimate' THEN 1 ELSE 0 END) as legitimate_count,
                AVG(inference_latency_ms) as avg_latency_ms,
                AVG(fraud_probability) as avg_fraud_probability,
                SUM(CASE WHEN user_label IS NOT NULL THEN 1 ELSE 0 END) as labeled_count,
                SUM(CASE WHEN user_label = prediction THEN 1 ELSE 0 END) as label_match_count
            FROM inference_log
        """)
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        stats = dict(zip(columns, row)) if row else {}
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Stats fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift-status")
def drift_status():
    if baselines is None:
        return {"status": "baselines_not_loaded", "drift_detected": False}
    return {
        "status": "ok",
        "baselines_loaded": True,
        "features_monitored": list(baselines.keys()),
    }


@app.get("/model-info")
def model_info():
    try:
        with open("data/production/metadata.json") as f:
            metadata = json.load(f)
        return {"model_info": metadata}
    except FileNotFoundError:
        return {"model_info": None, "message": "No production model metadata found"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", response_class=HTMLResponse)
def ui():
    with open("src/static/index.html", "r") as f:
        return f.read()