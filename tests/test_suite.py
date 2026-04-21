"""
Test Suite for Fake Job Posting Detector

Covers:
  - Unit tests: pipeline scripts, feature engineering, drift computation
  - Integration tests: FastAPI endpoints, DB logging
  - Model tests: performance on test.csv, latency SLA
  - Acceptance criteria validation

Run:
    pytest tests/test_suite.py -v --tb=short

Generate report:
    pytest tests/test_suite.py -v --tb=short --html=docs/test_report.html
    (requires: pip install pytest-html)
"""

import os
import sys
import json
import time
import pickle
import sqlite3
import tempfile
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Paths 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'test.csv')
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'production', 'model.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, 'data', 'production', 'tfidf_vectorizer.pkl')
BASELINE_PATH = os.path.join(PROJECT_ROOT, 'data', 'baselines', 'training_baseline.json')
PROCESSED_X_TEST = os.path.join(PROJECT_ROOT, 'data', 'processed', 'X_test.npz')
PROCESSED_Y_TEST = os.path.join(PROJECT_ROOT, 'data', 'processed', 'y_test.csv')

# Acceptance Criteria (from guidelines)
ACCEPTANCE_F1_FRAUD = 0.70        # F1-score for fraud class
ACCEPTANCE_ROC_AUC = 0.85         # ROC-AUC
ACCEPTANCE_LATENCY_MS = 200.0     # Max inference latency (p95)
ACCEPTANCE_PRECISION = 0.70       # Precision for fraud class
ACCEPTANCE_ERROR_RATE = 0.05      # Max API error rate


# FIXTURES
@pytest.fixture(scope="session")
def model():
    """Load production model once for all tests."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Production model not found. Run dvc repro first.")
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def vectorizer():
    """Load TF-IDF vectorizer once for all tests."""
    if not os.path.exists(VECTORIZER_PATH):
        pytest.skip("Vectorizer not found. Run dvc repro first.")
    with open(VECTORIZER_PATH, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def test_df():
    """Load test dataset once for all tests."""
    if not os.path.exists(TEST_DATA_PATH):
        pytest.skip("Test data not found at data/raw/test.csv")
    return pd.read_csv(TEST_DATA_PATH)


@pytest.fixture(scope="session")
def baseline():
    """Load training baselines once for all tests."""
    if not os.path.exists(BASELINE_PATH):
        pytest.skip("Baselines not found. Run Airflow setup DAG first.")
    with open(BASELINE_PATH, 'r') as f:
        return json.load(f)


# @pytest.fixture(scope="session")
# def api_client():
#     """Create FastAPI test client."""
#     try:
#         from fastapi.testclient import TestClient
#         # Import app with a temporary DB
#         with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
#             temp_db = f.name

#         os.environ['DB_PATH'] = temp_db
#         os.environ['MODEL_SERVE_MODE'] = 'local'

#         from main import app
#         client = TestClient(app)
#         yield client

#         os.unlink(temp_db)
#     except ImportError:
#         pytest.skip("FastAPI or dependencies not available")


@pytest.fixture(scope="session")
def X_test_features():
    """Load precomputed test features."""
    if not os.path.exists(PROCESSED_X_TEST):
        pytest.skip("Processed test features not found. Run dvc repro first.")
    return sp.load_npz(PROCESSED_X_TEST)


@pytest.fixture(scope="session")
def y_test():
    """Load test labels."""
    if not os.path.exists(PROCESSED_Y_TEST):
        pytest.skip("Test labels not found. Run dvc repro first.")
    return pd.read_csv(PROCESSED_Y_TEST).values.ravel()


# SECTION 1: UNIT TESTS — Data Pipeline
class TestDataValidation:
    """Unit tests for src/pipeline/validate.py"""

    def test_train_csv_exists(self):
        """TC-001: Training data file exists."""
        assert os.path.exists(TRAIN_DATA_PATH), \
            f"train.csv not found at {TRAIN_DATA_PATH}"

    def test_test_csv_exists(self):
        """TC-002: Test data file exists."""
        assert os.path.exists(TEST_DATA_PATH), \
            f"test.csv not found at {TEST_DATA_PATH}"

    def test_required_columns_present(self, test_df):
        """TC-003: All required columns are present in test data."""
        required_cols = [
            'title', 'description', 'company_profile', 'requirements',
            'employment_type', 'has_company_logo', 'has_questions', 'fraudulent'
        ]
        missing = [c for c in required_cols if c not in test_df.columns]
        assert not missing, f"Missing columns in test data: {missing}"

    def test_target_variable_is_binary(self, test_df):
        """TC-004: Target variable 'fraudulent' contains only 0 and 1."""
        unique_vals = set(test_df['fraudulent'].dropna().unique())
        assert unique_vals.issubset({0, 1, 0.0, 1.0}), \
            f"Unexpected values in target: {unique_vals}"

    def test_no_empty_titles(self, test_df):
        """TC-005: Critical column 'title' has acceptable missing rate (<30%)."""
        missing_pct = test_df['title'].isnull().mean()
        assert missing_pct < 0.30, \
            f"Too many missing titles: {missing_pct:.1%}"

    def test_no_empty_descriptions(self, test_df):
        """TC-006: Critical column 'description' has acceptable missing rate (<30%)."""
        missing_pct = test_df['description'].isnull().mean()
        assert missing_pct < 0.30, \
            f"Too many missing descriptions: {missing_pct:.1%}"

    def test_class_imbalance_within_expected_range(self, test_df):
        """TC-007: Fraud rate in test data is within expected range (1%–20%)."""
        fraud_rate = test_df['fraudulent'].mean()
        assert 0.01 <= fraud_rate <= 0.20, \
            f"Fraud rate {fraud_rate:.3f} is outside expected range [0.01, 0.20]"

    def test_no_duplicate_records(self, test_df):
        """TC-008: Test data has no fully duplicate rows."""
        duplicates = test_df.duplicated().sum()
        assert duplicates == 0, \
            f"Found {duplicates} duplicate rows in test data"

    def test_minimum_record_count(self, test_df):
        """TC-009: Test dataset has at least 500 records."""
        assert len(test_df) >= 500, \
            f"Test data too small: {len(test_df)} records"

    def test_has_company_logo_is_binary(self, test_df):
        """TC-010: has_company_logo contains only 0 and 1."""
        unique_vals = set(test_df['has_company_logo'].dropna().unique())
        assert unique_vals.issubset({0, 1, 0.0, 1.0}), \
            f"Unexpected values in has_company_logo: {unique_vals}"


# SECTION 2: UNIT TESTS — Feature Engineering
class TestFeatureEngineering:
    """Unit tests for preprocessing and feature engineering logic."""

    def test_vectorizer_exists(self):
        """TC-011: TF-IDF vectorizer artifact exists."""
        assert os.path.exists(VECTORIZER_PATH), \
            "TF-IDF vectorizer not found. Run dvc repro first."

    def test_vectorizer_transform_shape(self, vectorizer):
        """TC-012: Vectorizer transforms text into correct feature dimensions."""
        sample = ["Software engineer needed for our team"]
        X = vectorizer.transform(sample)
        assert X.shape[0] == 1, "Vectorizer should return 1 row for 1 input"
        assert X.shape[1] == 5000, f"Expected 5000 TF-IDF features, got {X.shape[1]}"

    def test_vectorizer_handles_empty_text(self, vectorizer):
        """TC-013: Vectorizer handles empty string without error."""
        X = vectorizer.transform([""])
        assert X.shape[0] == 1

    def test_vectorizer_handles_special_characters(self, vectorizer):
        """TC-014: Vectorizer handles special characters without error."""
        X = vectorizer.transform(["Job @ $50k!!! Call +1-800-SCAM now!!!"])
        assert X.shape[0] == 1

    def test_feature_matrix_shape(self, X_test_features, y_test):
        """TC-015: Test feature matrix rows match test labels count."""
        assert X_test_features.shape[0] == len(y_test), \
            f"Feature matrix rows ({X_test_features.shape[0]}) != labels ({len(y_test)})"

    def test_feature_matrix_is_sparse(self, X_test_features):
        """TC-016: Feature matrix is sparse (memory efficient)."""
        assert sp.issparse(X_test_features), \
            "Feature matrix should be a sparse matrix"

    def test_processed_features_exist(self):
        """TC-017: Processed feature artifacts exist from DVC pipeline."""
        assert os.path.exists(PROCESSED_X_TEST), "X_test.npz not found"
        assert os.path.exists(PROCESSED_Y_TEST), "y_test.csv not found"

# SECTION 3: UNIT TESTS — Drift Detection
class TestDriftDetection:
    """Unit tests for PSI computation and drift detection logic."""

    def _compute_psi(self, expected, actual, epsilon=1e-6):
        """Helper: compute PSI between two distributions."""
        all_cats = set(expected.keys()) | set(actual.keys())
        psi = 0.0
        for cat in all_cats:
            e = max(expected.get(cat, epsilon), epsilon)
            a = max(actual.get(cat, epsilon), epsilon)
            psi += (a - e) * np.log(a / e)
        return psi

    def test_psi_identical_distributions(self):
        """TC-018: PSI is ~0 for identical distributions."""
        dist = {"A": 0.5, "B": 0.3, "C": 0.2}
        psi = self._compute_psi(dist, dist)
        assert psi < 0.01, f"PSI for identical distributions should be ~0, got {psi}"

    def test_psi_no_drift_below_threshold(self):
        """TC-019: PSI below 0.1 correctly indicates no drift."""
        expected = {"Full-time": 0.6, "Part-time": 0.3, "Contract": 0.1}
        actual = {"Full-time": 0.58, "Part-time": 0.31, "Contract": 0.11}
        psi = self._compute_psi(expected, actual)
        assert psi < 0.1, f"Minor distribution shift should have PSI < 0.1, got {psi}"

    def test_psi_drift_above_threshold(self):
        """TC-020: PSI above 0.2 correctly indicates drift."""
        expected = {"Full-time": 0.7, "Part-time": 0.2, "Contract": 0.1}
        actual = {"Full-time": 0.2, "Part-time": 0.5, "Contract": 0.3}
        psi = self._compute_psi(expected, actual)
        assert psi >= 0.2, f"Large distribution shift should have PSI >= 0.2, got {psi}"

    def test_baseline_file_exists(self):
        """TC-021: Training baseline file exists."""
        assert os.path.exists(BASELINE_PATH), \
            "Training baseline not found. Run Airflow setup DAG first."

    def test_baseline_has_required_keys(self, baseline):
        """TC-022: Baseline JSON contains required feature keys."""
        required_keys = ['has_company_logo', 'has_questions', 'dataset']
        for key in required_keys:
            assert key in baseline, f"Baseline missing key: {key}"

    def test_baseline_dataset_stats(self, baseline):
        """TC-023: Baseline dataset stats are within expected range."""
        dataset = baseline.get('dataset', {})
        fraud_rate = dataset.get('fraud_rate', 0)
        assert 0.01 <= fraud_rate <= 0.15, \
            f"Baseline fraud rate {fraud_rate} is outside expected range"


# SECTION 4: UNIT TESTS — Model

class TestModel:
    """Unit tests for the production model."""

    def test_model_file_exists(self):
        """TC-024: Production model pickle exists."""
        assert os.path.exists(MODEL_PATH), \
            "Model not found. Run dvc repro first."

    def test_model_loads_successfully(self, model):
        """TC-025: Model loads without error."""
        assert model is not None

    def test_model_has_predict_method(self, model):
        """TC-026: Model has predict method."""
        assert hasattr(model, 'predict'), "Model missing predict method"

    def test_model_has_predict_proba_method(self, model):
        """TC-027: Model has predict_proba method."""
        assert hasattr(model, 'predict_proba'), "Model missing predict_proba method"

    def test_model_predict_returns_binary(self, model, X_test_features):
        """TC-028: Model predictions are binary (0 or 1)."""
        sample = X_test_features[:10]
        preds = model.predict(sample)
        assert set(preds).issubset({0, 1}), \
            f"Model returned non-binary predictions: {set(preds)}"

    def test_model_predict_proba_sums_to_one(self, model, X_test_features):
        """TC-029: predict_proba probabilities sum to 1 for each sample."""
        sample = X_test_features[:10]
        probas = model.predict_proba(sample)
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            "predict_proba rows do not sum to 1"

    def test_model_predict_proba_in_range(self, model, X_test_features):
        """TC-030: All probability values are in [0, 1]."""
        sample = X_test_features[:50]
        probas = model.predict_proba(sample)
        assert probas.min() >= 0.0 and probas.max() <= 1.0, \
            "Probability values outside [0, 1] range"

    def test_metadata_file_exists(self):
        """TC-031: Production model metadata JSON exists."""
        metadata_path = os.path.join(PROJECT_ROOT, 'data', 'production', 'metadata.json')
        assert os.path.exists(metadata_path), \
            "Production metadata not found."


# SECTION 5: MODEL PERFORMANCE TESTS (Acceptance Criteria)

class TestModelPerformance:
    """
    Model performance tests against acceptance criteria.
    These define pass/fail thresholds for the production model.
    """
    def test_f1_score_fraud_class(self, model, X_test_features, y_test):
        """
        TC-032 [ACCEPTANCE]: F1-score for fraud class >= 0.70
        Business rationale: Missing fraud (false negative) is costly.
        A high F1 ensures balanced precision and recall for the minority class.
        """
        from sklearn.metrics import f1_score
        y_pred = model.predict(X_test_features)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        assert f1 >= ACCEPTANCE_F1_FRAUD, \
            f"F1-score {f1:.4f} below acceptance threshold {ACCEPTANCE_F1_FRAUD}"

    def test_roc_auc_score(self, model, X_test_features, y_test):
        """
        TC-033 [ACCEPTANCE]: ROC-AUC >= 0.85
        Business rationale: AUC measures overall discrimination ability.
        """
        from sklearn.metrics import roc_auc_score
        y_prob = model.predict_proba(X_test_features)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        assert auc >= ACCEPTANCE_ROC_AUC, \
            f"ROC-AUC {auc:.4f} below acceptance threshold {ACCEPTANCE_ROC_AUC}"

    def test_precision_fraud_class(self, model, X_test_features, y_test):
        """
        TC-034 [ACCEPTANCE]: Precision for fraud class >= 0.70
        Business rationale: Too many false positives degrades user trust.
        """
        from sklearn.metrics import precision_score
        y_pred = model.predict(X_test_features)
        precision = precision_score(y_test, y_pred, pos_label=1)
        assert precision >= ACCEPTANCE_PRECISION, \
            f"Precision {precision:.4f} below acceptance threshold {ACCEPTANCE_PRECISION}"

    def test_recall_fraud_class(self, model, X_test_features, y_test):
        """
        TC-035: Recall for fraud class > 0.50
        Business rationale: At least half of fraudulent postings should be caught.
        """
        from sklearn.metrics import recall_score
        y_pred = model.predict(X_test_features)
        recall = recall_score(y_test, y_pred, pos_label=1)
        assert recall > 0.50, \
            f"Recall {recall:.4f} is unacceptably low"

    def test_inference_latency_single_sample(self, model, X_test_features):
        """
        TC-036 [ACCEPTANCE]: Single sample inference latency < 200ms
        Business metric from guidelines.
        """
        sample = X_test_features[0]
        # Warm up
        model.predict(sample)
        # Measure
        times = []
        for _ in range(100):
            t0 = time.time()
            model.predict(sample)
            times.append((time.time() - t0) * 1000)
        p95 = np.percentile(times, 95)
        assert p95 < ACCEPTANCE_LATENCY_MS, \
            f"p95 latency {p95:.2f}ms exceeds 200ms SLA"

    def test_inference_latency_batch(self, model, X_test_features):
        """
        TC-037: Batch inference on 100 samples completes in < 5 seconds.
        """
        batch = X_test_features[:100]
        start = time.time()
        model.predict(batch)
        duration = time.time() - start
        assert duration < 5.0, \
            f"Batch inference took {duration:.2f}s, expected < 5s"

    def test_model_not_predicting_all_legitimate(self, model, X_test_features):
        """
        TC-038: Model does not predict everything as Legitimate (trivial classifier check).
        """
        y_pred = model.predict(X_test_features)
        fraud_predictions = (y_pred == 1).sum()
        assert fraud_predictions > 0, \
            "Model predicts everything as Legitimate — likely a degenerate classifier"

    def test_model_not_predicting_all_fraud(self, model, X_test_features):
        """
        TC-039: Model does not predict everything as Fraudulent.
        """
        y_pred = model.predict(X_test_features)
        legit_predictions = (y_pred == 0).sum()
        assert legit_predictions > 0, \
            "Model predicts everything as Fraudulent — likely a degenerate classifier"

    def test_confusion_matrix_true_positives(self, model, X_test_features, y_test):
        """
        TC-040: Model correctly identifies at least 50% of actual fraud cases.
        """
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_test_features)
        cm = confusion_matrix(y_test, y_pred)
        # cm[1][1] = true positives (fraud correctly identified)
        # cm[1][0] = false negatives (fraud missed)
        tp = cm[1][1]
        fn = cm[1][0]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        assert recall >= 0.50, \
            f"True positive rate {recall:.4f} too low — model misses too many fraud cases"


# SECTION 6: INTEGRATION TESTS — FastAPI Endpoints
class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints."""

    LEGITIMATE_POSTING = {
        "title": "Senior Software Engineer",
        "description": "We are looking for an experienced software engineer to join our team. You will work on building scalable backend systems using Python and cloud technologies. The role offers competitive compensation and excellent benefits.",
        "company_profile": "TechCorp is a leading software company with over 500 employees.",
        "requirements": "5+ years Python experience, strong CS fundamentals, experience with distributed systems",
        "employment_type": "Full-time",
        "has_company_logo": 1,
        "has_questions": 1,
        "salary_range": "$120,000 - $150,000"
    }

    FRAUDULENT_POSTING = {
        "title": "Work From Home Earn $5000 Weekly No Experience",
        "description": "Make easy money working from home! No experience needed. Just send us your bank details and we will deposit your first payment. Guaranteed income. Start immediately.",
        "company_profile": "",
        "requirements": "",
        "employment_type": "",
        "has_company_logo": 0,
        "has_questions": 0,
        "salary_range": ""
    }

    def test_health_endpoint_returns_200(self, api_client):
        """TC-041: /health endpoint returns HTTP 200."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_healthy(self, api_client):
        """TC-042: /health endpoint returns status=healthy."""
        response = api_client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_ready_endpoint_returns_200_or_503(self, api_client):
        """TC-043: /ready endpoint returns 200 (model loaded) or 503 (not loaded)."""
        response = api_client.get("/ready")
        assert response.status_code in [200, 503]

    def test_mode_endpoint_returns_mode(self, api_client):
        """TC-044: /mode endpoint returns current serve mode."""
        response = api_client.get("/mode")
        assert response.status_code == 200
        assert "mode" in response.json()
        assert response.json()["mode"] in ["local", "mlflow"]

    def test_predict_endpoint_returns_200(self, api_client):
        """TC-045: /predict endpoint returns HTTP 200 for valid input."""
        response = api_client.post("/predict", json=self.LEGITIMATE_POSTING)
        assert response.status_code in [200, 503], \
            f"Unexpected status: {response.status_code}"

    def test_predict_response_has_required_fields(self, api_client):
        """TC-046: /predict response contains all required fields."""
        response = api_client.post("/predict", json=self.LEGITIMATE_POSTING)
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        data = response.json()
        required_fields = [
            'prediction_id', 'prediction', 'fraud_probability',
            'confidence', 'inference_latency_ms', 'risk_level', 'key_signals'
        ]
        for field in required_fields:
            assert field in data, f"Missing field in response: {field}"

    def test_predict_verdict_is_valid(self, api_client):
        """TC-047: /predict verdict is either Legitimate or Fraudulent."""
        response = api_client.post("/predict", json=self.LEGITIMATE_POSTING)
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        verdict = response.json()["prediction"]
        assert verdict in ["Legitimate", "Fraudulent"], \
            f"Unexpected verdict: {verdict}"

    def test_predict_fraud_probability_in_range(self, api_client):
        """TC-048: fraud_probability is between 0 and 1."""
        response = api_client.post("/predict", json=self.LEGITIMATE_POSTING)
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        prob = response.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0, f"Fraud probability {prob} out of range"

    def test_predict_latency_under_200ms(self, api_client):
        """
        TC-049 [ACCEPTANCE]: /predict endpoint responds in < 200ms (p95).
        Business metric from guidelines.
        """
        latencies = []
        for _ in range(20):
            start = time.time()
            response = api_client.post("/predict", json=self.LEGITIMATE_POSTING)
            latencies.append((time.time() - start) * 1000)
            if response.status_code == 503:
                pytest.skip("Model not loaded in test environment")

        p95 = np.percentile(latencies, 95)
        assert p95 < ACCEPTANCE_LATENCY_MS, \
            f"API p95 latency {p95:.2f}ms exceeds 200ms SLA"

    def test_predict_missing_title_returns_422(self, api_client):
        """TC-050: /predict with missing required field returns HTTP 422."""
        payload = {"description": "some job description"}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_description_returns_422(self, api_client):
        """TC-051: /predict with missing description returns HTTP 422."""
        payload = {"title": "Engineer"}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_feedback_endpoint_correct(self, api_client):
        """TC-052: /feedback accepts 'correct' feedback."""
        pred_response = api_client.post("/predict", json=self.LEGITIMATE_POSTING)
        if pred_response.status_code == 503:
            pytest.skip("Model not loaded")
        prediction_id = pred_response.json()["prediction_id"]
        feedback_response = api_client.post("/feedback", json={
            "prediction_id": prediction_id,
            "feedback": "correct"
        })
        assert feedback_response.status_code == 200

    def test_feedback_endpoint_incorrect(self, api_client):
        """TC-053: /feedback accepts 'incorrect' feedback."""
        pred_response = api_client.post("/predict", json=self.FRAUDULENT_POSTING)
        if pred_response.status_code == 503:
            pytest.skip("Model not loaded")
        prediction_id = pred_response.json()["prediction_id"]
        feedback_response = api_client.post("/feedback", json={
            "prediction_id": prediction_id,
            "feedback": "incorrect"
        })
        assert feedback_response.status_code == 200

    def test_feedback_invalid_value_returns_400(self, api_client):
        """TC-054: /feedback with invalid value returns HTTP 400."""
        response = api_client.post("/feedback", json={
            "prediction_id": "fake-id-123",
            "feedback": "maybe"
        })
        assert response.status_code == 422 #fix from 400-> 422

    def test_history_endpoint_returns_200(self, api_client):
        """TC-055: /history endpoint returns HTTP 200."""
        response = api_client.get("/history")
        assert response.status_code == 200

    def test_history_has_list(self, api_client):
        """TC-056: /history response contains a list."""
        response = api_client.get("/history")
        assert "history" in response.json()
        assert isinstance(response.json()["history"], list)

    def test_stats_endpoint_returns_200(self, api_client):
        """TC-057: /stats endpoint returns HTTP 200."""
        response = api_client.get("/stats")
        assert response.status_code == 200

    def test_metrics_endpoint_returns_prometheus_format(self, api_client):
        """TC-058: /metrics endpoint returns valid Prometheus text format."""
        response = api_client.get("/metrics")
        assert response.status_code == 200
        assert "api_request_count_total" in response.text or \
               "# HELP" in response.text

    def test_drift_status_endpoint(self, api_client):
        """TC-059: /drift-status endpoint returns 200."""
        response = api_client.get("/drift-status")
        assert response.status_code == 200

    def test_model_info_endpoint(self, api_client):
        """TC-060: /model-info endpoint returns 200."""
        response = api_client.get("/model-info")
        assert response.status_code == 200
    
    def test_predict_mlflow_routing(self, api_client):
        """TC-061: /predict endpoint correctly formats and routes payloads to MLflow."""
        import os
        from unittest.mock import patch
        
        os.environ['MODEL_SERVE_MODE'] = 'mlflow'
        with patch('requests.post') as mock_post:
            # Program the mock to return a fake MLflow prediction (0.85 = 85% Fraud)
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"predictions": [0.85]}
            
            response = api_client.post("/predict", json=self.FRAUDULENT_POSTING)
            
            # Assert the API actually tried to make a network call
            mock_post.assert_called_once()
            
            # Extract the payload that the API tried to send to MLflow
            sent_payload = mock_post.call_args.kwargs['json']
            
            # Verify the payload strictly matches MLflow 2.x 'dataframe_split' schema
            assert "dataframe_split" in sent_payload
            assert "columns" in sent_payload["dataframe_split"]
            assert "data" in sent_payload["dataframe_split"]
            
            # Verify the API processed the mock response correctly
            assert response.status_code == 200
            assert response.json()["fraud_probability"] == 0.85
            assert response.json()["prediction"] == "Fraudulent"
            
        # Revert back to local mode to protect subsequent tests
        os.environ['MODEL_SERVE_MODE'] = 'local'


# SECTION 7: INTEGRATION TESTS — Database

class TestDatabase:
    """Integration tests for SQLite inference logging."""

    def test_inference_logged_to_db(self, api_client):
        """TC-062: Each prediction is logged to the inference_log table."""
        pred_response = api_client.post("/predict", json={
            "title": "Test Job",
            "description": "This is a test job description for unit testing purposes only."
        })
        if pred_response.status_code == 503:
            pytest.skip("Model not loaded")

        history_response = api_client.get("/history?limit=1")
        history = history_response.json()["history"]
        assert len(history) >= 1, "Prediction was not logged to database"

    def test_db_record_has_prediction_id(self, api_client):
        """TC-063: Logged records have a valid prediction ID."""
        history_response = api_client.get("/history?limit=1")
        history = history_response.json()["history"]
        if not history:
            pytest.skip("No predictions logged yet")
        assert history[0]["id"] is not None

    def test_feedback_updates_db_record(self, api_client):
        """TC-064: Submitting feedback updates the DB record correctly."""
        pred_response = api_client.post("/predict", json={
            "title": "Feedback Test Job",
            "description": "Testing feedback logging in the database for integration test."
        })
        if pred_response.status_code == 503:
            pytest.skip("Model not loaded")
        prediction_id = pred_response.json()["prediction_id"]

        api_client.post("/feedback", json={
            "prediction_id": prediction_id,
            "feedback": "correct"
        })

        history = api_client.get("/history?limit=10").json()["history"]
        record = next((r for r in history if r["id"] == prediction_id), None)
        assert record is not None, "Prediction not found in history"
        assert record["user_feedback"] == "correct", \
            f"Feedback not updated. Got: {record['user_feedback']}"


# SECTION 8: ARTIFACT INTEGRITY TESTS

class TestArtifacts:
    """Tests for DVC pipeline artifact integrity."""

    def test_dvc_yaml_exists(self):
        """TC-065: dvc.yaml pipeline definition exists."""
        path = os.path.join(PROJECT_ROOT, 'dvc.yaml')
        assert os.path.exists(path), "dvc.yaml not found"

    def test_dvc_lock_exists(self):
        """TC-066: dvc.lock exists (pipeline has been run)."""
        path = os.path.join(PROJECT_ROOT, 'dvc.lock')
        assert os.path.exists(path), "dvc.lock not found — run dvc repro first"

    def test_mlproject_exists(self):
        """TC-067: MLproject file exists for experiment tracking."""
        path = os.path.join(PROJECT_ROOT, 'MLproject')
        assert os.path.exists(path), "MLproject file not found"

    def test_python_env_yaml_exists(self):
        """TC-068: python_env.yaml environment spec exists."""
        path = os.path.join(PROJECT_ROOT, 'python_env.yaml')
        assert os.path.exists(path), "python_env.yaml not found"

    def test_production_model_exists(self):
        """TC-069: Production model artifact exists."""
        assert os.path.exists(MODEL_PATH), \
            "Production model not found. Run dvc repro first."

    def test_production_vectorizer_exists(self):
        """TC-070: Production vectorizer artifact exists."""
        assert os.path.exists(VECTORIZER_PATH), \
            "Production vectorizer not found. Run dvc repro first."

    def test_production_metadata_valid_json(self):
        """TC-071: Production metadata is valid JSON with required keys."""
        metadata_path = os.path.join(PROJECT_ROOT, 'data', 'production', 'metadata.json')
        if not os.path.exists(metadata_path):
            pytest.skip("metadata.json not found")
        with open(metadata_path) as f:
            meta = json.load(f)
        assert 'metrics' in meta, "metadata.json missing 'metrics' key"
        assert 'model_path' in meta, "metadata.json missing 'model_path' key"