"""
Fake Job Posting Detector — FastAPI Backend
Handles prediction, feedback logging, encrypted inference storage, and monitoring.
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
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import sqlite3
from cryptography.fernet import Fernet
import requests as req
# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# --- API Health Dashboard Metrics ---
REQUEST_COUNT = Counter("api_request_count", "Total API requests", ["endpoint", "method", "status"])
ERROR_COUNT = Counter("api_error_count", "Total API errors")
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"], buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
SLA_BREACH_COUNT = Counter("api_sla_breach_count", "API requests exceeding SLA")
API_UPTIME = Gauge("api_uptime_seconds", "Total seconds API has been running")

# --- Model Performance Dashboard Metrics ---
PREDICTION_COUNT = Counter("model_prediction_count", "Total predictions made", ["prediction"])
FRAUD_RATE = Gauge("model_fraud_rate_current", "Current running average of fraud predictions")
PREDICTION_CONFIDENCE = Histogram("model_prediction_confidence", "Model prediction confidence scores", buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
INFERENCE_LATENCY = Histogram("model_inference_latency_ms", "Actual model scoring latency", buckets=[10, 50, 100, 200, 500, 1000])
SERVE_MODE_GAUGE = Gauge("model_serve_mode", "Current active model backend (0=MLflow, 1=Local)")
USER_FEEDBACK = Counter("user_feedback_count", "User feedback submitted via UI", ["feedback_type"])

# App Setup

# App Setup
app = FastAPI(title="Fake Job Posting Detector", description="Detects fraudulent job postings using ML", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Nginx frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config & Encryption Setup
MODEL_PATH = os.getenv("MODEL_PATH", "data/production/model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "data/production/tfidf_vectorizer.pkl")
BASELINE_PATH = os.getenv("BASELINE_PATH", "baselines/training_baseline.json") # Updated for volume mount
MODEL_SERVE_MODE = os.getenv("MODEL_SERVE_MODE", "local")
MLFLOW_SERVE_URL = os.getenv("MLFLOW_SERVE_URL", "http://mlflow_model:5001/invocations")

# Use Airflow's Fernet key for encryption, or generate a fallback for testing
FERNET_KEY = os.getenv("AIRFLOW__CORE__FERNET_KEY")
if not FERNET_KEY:
    FERNET_KEY = Fernet.generate_key().decode()
    logger.warning("AIRFLOW__CORE__FERNET_KEY not found. Using ephemeral key. Data will be lost on restart.")

cipher_suite = Fernet(FERNET_KEY.encode())

def encrypt_data(text: Optional[str]) -> Optional[str]:
    if not text: return text
    return cipher_suite.encrypt(str(text).encode()).decode()

def decrypt_data(text: Optional[str]) -> Optional[str]:
    if not text: return text
    try:
        return cipher_suite.decrypt(str(text).encode()).decode()
    except:
        return text # Fallback for unencrypted legacy data

# Global State
model = None
vectorizer = None
baselines = None

# Database Setup
def get_db():
    return sqlite3.connect("data/inference.db")

def init_db():
    try:
        os.makedirs("data", exist_ok=True)
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inference_log (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, title TEXT,
                description TEXT, company_profile TEXT, requirements TEXT,
                employment_type TEXT, has_company_logo INTEGER, has_questions INTEGER,
                salary_range TEXT, prediction TEXT NOT NULL, fraud_probability REAL NOT NULL,
                confidence REAL NOT NULL, inference_latency_ms REAL, user_label TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# Model Loading
def load_model():
    global model, vectorizer, baselines
    try:
        with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
        logger.info("Model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Model file not found: {e}")

    try:
        with open(BASELINE_PATH, "r") as f: baselines = json.load(f)
    except FileNotFoundError:
        logger.warning("Baselines file not found — drift detection disabled.")

@app.on_event("startup")
async def startup():
    init_db()
    load_model()

# Schemas
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

class FeedbackSchema(BaseModel):
    prediction_id: str
    correct_label: str

# Helper: Feature Engineering (matches your preprocess.py)
def build_features(posting: JobPosting):
    combined_text = " ".join([
        posting.title or "", posting.location or "", posting.description or "",
        posting.company_profile or "", posting.requirements or "", posting.benefits or "",
    ])
    has_salary = 1 if posting.salary_range else 0
    has_company_profile = 1 if (posting.company_profile and posting.company_profile.strip()) else 0
    has_requirements = 1 if (posting.requirements and posting.requirements.strip()) else 0
    has_benefits = 1 if (posting.benefits and posting.benefits.strip()) else 0

    X_text = vectorizer.transform([combined_text])
    X_meta = sp.csr_matrix([[
        int(posting.has_company_logo or 0), int(posting.has_questions or 0),
        has_salary, has_company_profile, has_requirements,
        int(posting.telecommuting or 0), has_benefits,
    ]])
    return sp.hstack([X_text, X_meta])

def get_risk_level(prob: float) -> str:
    if prob < 0.3: return "LOW"
    elif prob < 0.6: return "MEDIUM"
    elif prob < 0.8: return "HIGH"
    return "CRITICAL"

def get_key_signals(posting: JobPosting, prob: float) -> list:
    signals = []
    if not posting.company_profile: signals.append("No company profile provided")
    if not posting.has_company_logo: signals.append("No company logo")
    if not posting.salary_range: signals.append("Salary range not disclosed")
    if not posting.requirements: signals.append("No requirements listed")
    if not posting.has_questions: signals.append("No screening questions")
    if len(posting.description) < 100: signals.append("Very short job description")
    if prob > 0.5 and not signals: signals.append("Suspicious text patterns detected")
    return signals[:4]

# Endpoints
@app.get("/health")
def health(): return {"status": "healthy"}

@app.get("/ready")
def ready():
    if MODEL_SERVE_MODE == "local" and (model is None or vectorizer is None):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "mode": MODEL_SERVE_MODE}

@app.get("/mode")
def get_mode():
    return {"mode": MODEL_SERVE_MODE, "mlflow_url": MLFLOW_SERVE_URL if MODEL_SERVE_MODE == "mlflow" else None}

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
        # Encrypting sensitive text fields before DB insertion
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
            encrypt_data(posting.title), encrypt_data(posting.description), 
            encrypt_data(posting.company_profile), encrypt_data(posting.requirements), 
            posting.employment_type, posting.has_company_logo, posting.has_questions,
            encrypt_data(posting.salary_range), prediction,
            fraud_prob, confidence, latency_ms, encrypt_data(posting.user_label)
        ))
        conn.commit()
        conn.close()
    except Exception as db_err:
        logger.error(f"Failed to log to DB: {db_err}")

    return PredictionResponse(
        prediction_id=prediction_id, prediction=prediction, fraud_probability=round(fraud_prob, 4),
        confidence=round(confidence, 4), inference_latency_ms=round(latency_ms, 2),
        risk_level=get_risk_level(fraud_prob), key_signals=get_key_signals(posting, fraud_prob)
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(posting: JobPosting):
    REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
    start = time.time()
    try:
        # 1. Transform text to TF-IDF numerical features FIRST, regardless of mode
        X = build_features(posting)
        
        if MODEL_SERVE_MODE == "mlflow":
            import requests as req
            
            # 2. Convert the TF-IDF sparse matrix/array to a native Python list
            inputs_list = X.toarray().tolist() if hasattr(X, "toarray") else X.tolist()
            
            # 3. Send the numerical array to MLflow
            payload = {"inputs": inputs_list}
            res = req.post(MLFLOW_SERVE_URL, json=payload, timeout=5)
            
            # CRITICAL: Expose the real MLflow error if it fails!
            if res.status_code != 200:
                raise Exception(f"MLflow Server Error ({res.status_code}): {res.text}")
            
            # 4. Parse the prediction
            prediction_result = res.json()["predictions"][0]
            
            # Note: A raw MLflow sklearn model returns predict() [e.g., 0 or 1],
            # rather than predict_proba() unless specifically configured.
            if isinstance(prediction_result, list):
                fraud_prob = float(prediction_result[1])
            else:
                fraud_prob = float(prediction_result)
                
        else:
            if model is None: raise HTTPException(status_code=503, detail="Local model not loaded")
            fraud_prob = float(model.predict_proba(X)[0][1])
        
        pred_label = "Fraudulent" if fraud_prob >= 0.5 else "Legitimate"
        PREDICTION_COUNT.labels(prediction=pred_label).inc()
        confidence = fraud_prob if fraud_prob >= 0.5 else (1.0 - fraud_prob)
        PREDICTION_CONFIDENCE.observe(confidence)
        latency_ms = (time.time() - start) * 1000
        INFERENCE_LATENCY.observe(latency_ms)
            
        return build_response(posting, fraud_prob, time.time() - start)
        
    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        # This will now print the actual MLflow error text to the UI/logs!
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def submit_feedback(feedback: FeedbackSchema):
    """Updates the user_label for an existing prediction."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "UPDATE inference_log SET user_label = ? WHERE id = ?",
            (encrypt_data(feedback.correct_label), feedback.prediction_id)
        )
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@app.get("/history")
def get_history(limit: int = 20):
    try:
        conn = get_db()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, timestamp, title, prediction, fraud_probability, confidence, inference_latency_ms, user_label FROM inference_log ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        
        # Decrypt fields for the frontend
        history = []
        for r in rows:
            d = dict(r)
            d['title'] = decrypt_data(d['title'])
            d['user_label'] = decrypt_data(d['user_label'])
            history.append(d)
            
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) as total_predictions,
                   SUM(CASE WHEN prediction = 'Fraudulent' THEN 1 ELSE 0 END) as fraud_count,
                   AVG(inference_latency_ms) as avg_latency_ms
            FROM inference_log
        """)
        row = cur.fetchone()
        
        # Manual label matching count required because DB labels are encrypted now
        cur.execute("SELECT prediction, user_label FROM inference_log WHERE user_label IS NOT NULL")
        labeled_rows = cur.fetchall()
        conn.close()

        labeled_count = 0
        label_match_count = 0
        for pred, enc_label in labeled_rows:
            if enc_label:
                labeled_count += 1
                if pred == decrypt_data(enc_label):
                    label_match_count += 1

        stats = {
            "total_predictions": row[0] or 0,
            "fraud_count": row[1] or 0,
            "avg_latency_ms": row[2],
            "labeled_count": labeled_count,
            "label_match_count": label_match_count
        }
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)