# Low-Level Design (LLD): Fake Job Postings Detector

## Overview
This document breaks down the internal wiring of our MLOps pipeline. We've built a system that bridges the gap between local experimentation and automated continuous training, keeping our dependencies lean and our serving layer highly flexible. Since we are targeting environments that might not have GPU access, the architecture leans heavily on efficient, CPU-friendly microservices and algorithms.

## 1. System Architecture & Topology
We're orchestrating the entire stack through Docker Compose to isolate our environments cleanly:
* **PostgreSQL:** The backbone for both MLflow and Airflow tracking.
* **MLflow Stack:** Split across two containers. The `mlflow` service runs the tracking UI (port 5000), while `mlflow_model` exposes the actual model inference endpoint (port 5001).
* **API Backend:** A FastAPI gateway (port 8000) that handles prediction routing, data encryption, and telemetry.
* **Monitoring:** We use Prometheus to scrape API metrics, Grafana for visualization, and Alertmanager to ping us via SMTP when drift occurs or SLAs breach.
* **Airflow:** Handles the heavy lifting for our Continuous Training and Daily Drift checks via a `LocalExecutor`.

## 2. Data Engineering Pipeline (DVC)
We strictly version our data transformations using DVC to ensure reproducible state across runs.

### 2.1. Splitting & Stratification
* **Logic:** The raw `fake_job_postings.csv` undergoes a standard 80/20 train-test split. 
* **Imbalance Handling:** Since fraudulent jobs are rare (~4.3%), we stratify the split on the target variable to guarantee proportional representation in both sets.

### 2.2. Feature Engineering
* **Text Processing:** We concatenate the core text fields (title, description, profile, etc.) into a single block. TF-IDF vectorization is capped at 5000 features with a (1, 2) n-gram range to keep memory usage in check.
* **Metadata Extraction:** We extract 7 explicit binary flags (e.g., `has_salary`, `telecommuting`).
* **Matrix Assembly:** The TF-IDF text matrix and metadata are horizontally stacked into a highly efficient `scipy.sparse.csr_matrix`. The fitted vectorizer is pickled to ensure exact state replication during inference.

### 2.3. Drift Baselines
* A dedicated script calculates mathematical baselines (means, standard deviations, distributions) for our features. This outputs `training_baseline.json`, which Airflow later consumes to calculate Population Stability Index (PSI) drift scores.

## 3. The Unified Training Engine
Our `train.py` script uses a "Y-Shaped" architecture to serve two masters: local DVC runs and automated Airflow pipelines.

* **Core Algorithm:** We went with a `GradientBoostingClassifier`—it's highly performant on tabular/sparse data and trains reliably on standard CPU hardware without needing a GPU cluster.
* **Strict Dependency Control:** To keep our calculation logic tight and predictable, we intentionally stripped out NumPy for all standard aggregations (like calculating sample weights and average latency). Instead, we enforce a gentle 5.0 penalty on fraudulent samples using pure Python list comprehensions.
* **Artifact Proxying:** When Airflow passes the `--use-mlflow` flag, the script forces `artifact_location="mlflow-artifacts:/"`. This brilliant workaround bypasses Docker volume permission errors, streaming our metrics and confusion matrices directly over HTTP to the MLflow server.

## 4. Orchestration (Airflow DAGs)
* Airflow acts as the brain of the automated MLOps pipeline, running two primary DAGs:

### 4.1. Daily Drift Monitoring (fake_job_drift_monitoring)
* **Trigger:** Runs `@daily`.

* **Flow:** Fetches the last 24 hours of decrypted inference logs from SQLite -> Computes Population Stability Index (PSI) against training_baseline.json.

* **Branching:** If drift is detected, it builds an HTML alert email and sends it via SMTP gmail. If the system is healthy, it sends a clean heartbeat summary email instead.

### 4.2. Continuous Training (fake_job_retraining_pipeline)
* **Trigger:** Manual (upon receiving a drift alert). Takes hyperparameters as input.

* **Flow:** Extracts user-labeled logs -> Decrypts data -> Merges with root training data -> Bypasses DVC locks to swap data -> Runs dvc repro -f up to feature engineering.

* **MLflow Handoff:** Executes the training script with `--use-mlflow`, streaming the new model directly to the jobguard-automated-retraining-exp registry.

* **Human-in-the-Loop:** Pings the admin with an email detailing the new F1/AUC scores and instructions for running the final dvc push locally upon approval.

## 5. The Unified Training Engine
The train.py script serves two masters: local DVC execution and MLflow automated runs.

* **Core Algorithm:** Uses a GradientBoostingClassifier, perfect for sparse TF-IDF matrices on CPU hardware.

* **Calculation Constraints:** We stripped out NumPy for core aggregations. Fraudulent samples are given a 5.0 penalty using pure Python list comprehensions.

* **Artifact Proxying:** When Airflow runs it with --use-mlflow, the script intercepts the MLflow client and injects `artifact_location="mlflow-artifacts:/"`. This forces artifacts to stream over the Docker HTTP network, entirely bypassing the container volume permission errors.

## 6. Serving Layer & API Contracts
The FastAPI backend (main.py) handles intelligent routing, privacy, and telemetry.

* **Dynamic Routing:** A `MODEL_SERVE_MODE` environment variable allows the /predict endpoint to instantly swap between reading model.pkl locally or querying the MLflow serving container (`http://mlflow_model:5001/invocations`) via HTTP.

* **Data Encryption:** All PII hitting the inference_log SQLite database is symmetrically encrypted using Fernet keys injected by Airflow.

* **Endpoints:**  
    - `POST /predict`: Returns fraud probability, confidence, risk levels (LOW to CRITICAL), and interpretable key signals.

    - `POST /feedback`: Captures ground-truth user corrections.

## 7. Monitoring & Alerting (Prometheus/Alertmanager)
We defined strict PromQL alerting rules in `alert_rules.yml` to catch both infrastructure and model degradation in real-time.

### 7.1. API & Performance Alerts
* **HighErrorRate:** Triggers CRITICAL if the API returns 500s for more than 5% of requests over a 5-minute window.

* **HighInferenceLatency:** Triggers WARNING if the 95th percentile (p95) latency exceeds our strict 200ms SLA over 5 minutes.

* **SLABreachRateHigh:** Alerts if SLA breaches exceed 0.1 per second.

### 7.2. Drift & Model Behavior Alerts
* **DataDriftDetected:** Triggers a WARNING if any feature's PSI score exceeds 0.2 for 5 minutes. Upgrades to CRITICAL (DriftConfirmed) if sustained for 10 minutes.

* **FraudRateAnomaly:** Triggers if the live model's prediction rate deviates by more than 15% absolute from the baseline 4.8%.

* **LowPredictionConfidence:** Triggers if the median model confidence drops below 60%, a strong signal of concept drift.

* **NoRecentPredictions:** Fires an INFO alert if 30 minutes pass without a single API hit, ensuring we know if the frontend disconnected.

## 8. Architecture Diagram:


```text
+-----------------------------------------------------------------------------------------------------------+
|                                      Fake Job Detector - Architecture                                     |
+-----------------------------------------------------------------------------------------------------------+

                                    [ End Users / UI ]
                                          | (HTTP POST /predict)
                                          v
+----------------------+         +----------------------------------+        +-----------------------------+
|         Local        |<========|       FastAPI Backend            |=======>|   MLflow Model Serving      |
|   (/data/production) |         | (Routing, Telemetry, Privacy)    |        |   (Port 5001 - @Production) |<--+
+----------------------+         +----------------------------------+        +-----------------------------+   |
            ^                        |                   |                                                     |
            |                  (Encrypted Logs)    (Scrape Metrics)                                            |
            |                        |                   |                                                     |
            |                        v                   v                                                     |
            |                  +-------------+   +----------------+        +---------------------------+       |   
            |                  | SQLite DB   |   | Prometheus     |=======>| Grafana / Alertmanager    |       |(Model
            |                  |             |   |                |        |                           |       |  Registry)
            |                  | (Inference) |   | (Time-series)  |        | (Dashboards & Alerts)     |       |
            |                  +-------------+   +----------------+        +---------------------------+       |
            |                        |                                                                         |
            |                  (Airflow Extracts Labeled Logs)                                                 |
            |                        |                                                                         |
            |                        v                                                                         |
            |                  +----------------------------------+        +---------------------------+       |
            |                  |      Airflow Orchestrator        |=======>| MLflow Tracking Server    |_______|
            |                  | (Data Prep, DVC Repro, Training) |        | (Port 5000 - Experiments) |
            |                  +----------------------------------+        +---------------------------+
            |                        |                   |
            |                  (State & Data)      (Metadata)
            |                        |                   |
            |                        v                   v
            |                  +-------------+   +-----------------+
            |__________________| Local Files |   | PostgreSQL      |
                               | (DVC / Git) |   | (Airflow/MLflow)|
                              +-------------+   +-----------------+
                              ```