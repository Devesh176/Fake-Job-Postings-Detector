# Test Plan — Fake Job Posting Detector

**Project:** Fake Job Posting Detector (MLOps Course Project)

---

## 1. Scope

This test plan covers all testable components of the Fake Job Posting Detector system including data pipelines, ML model performance, API endpoints, database logging, and artifact integrity.

---

## 2. Test Objectives

- Validate that all data pipeline outputs meet quality requirements
- Confirm the production model meets acceptance criteria on held-out test data
- Verify all API endpoints return correct responses with valid schemas
- Ensure inference logging works correctly end-to-end
- Validate all DVC and MLflow artifacts are present and valid

---

## 3. Acceptance Criteria

These criteria define the pass/fail threshold for the overall system:

| ID | Criterion | Threshold | Priority |
|----|-----------|-----------|----------|
| AC-001 | F1-Score (Fraud class) on test.csv | ≥ 0.70 | Critical |
| AC-002 | ROC-AUC on test.csv | ≥ 0.85 | Critical |
| AC-003 | Precision (Fraud class) on test.csv | ≥ 0.70 | High |
| AC-004 | Inference latency p95 (single sample) | < 200ms | Critical |
| AC-005 | API error rate | < 5% | Critical |
| AC-006 | API /health returns 200 | Always | Critical |
| AC-007 | Model does not predict trivially | Non-degenerate | High |

---

## 4. Test Categories

### 4.1 Unit Tests — Data Pipeline (TC-001 to TC-010)
Validates raw and test data quality, schema correctness, and class distribution.

### 4.2 Unit Tests — Feature Engineering (TC-011 to TC-017)
Validates TF-IDF vectorizer behavior, feature matrix shape and sparsity.

### 4.3 Unit Tests — Drift Detection (TC-018 to TC-023)
Validates PSI computation logic and baseline file integrity.

### 4.4 Unit Tests — Model (TC-024 to TC-031)
Validates model artifact existence, interface, and basic output validity.

### 4.5 Model Performance Tests (TC-032 to TC-040)
Validates model against acceptance criteria on held-out test.csv.

### 4.6 Integration Tests — API (TC-041 to TC-060)
Validates all FastAPI endpoints for correct behavior, schema, and latency.

### 4.7 Integration Tests — Database (TC-061 to TC-063)
Validates SQLite inference logging and feedback update.

### 4.8 Artifact Integrity Tests (TC-064 to TC-070)
Validates DVC and MLflow artifacts exist and are well-formed.

---

## 5. Test Execution

```bash
# Run all tests
pytest tests/test_suite.py -v --tb=short

# Run with report generation
python tests/run_tests.py

# Run specific category
pytest tests/test_suite.py::TestModelPerformance -v
```

---

## 6. Pass/Fail Definition

The test suite **passes** if:
- All acceptance criteria tests (AC-001 to AC-007) pass
- No integration tests fail
- Skipped tests are justified by environment constraints

The test suite **fails** if:
- Any acceptance criteria test fails
- Any API endpoint returns unexpected status codes
- Model makes trivially wrong predictions