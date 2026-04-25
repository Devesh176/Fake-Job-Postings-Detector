# High-Level Design (HLD): JobGuard (Fake Job Postings Detector)

## 1. Executive Summary
This an end-to-end Machine Learning Operations (MLOps) platform designed to detect fraudulent job postings in real-time. Rather than just deploying a static model, this system implements a fully containerized Continuous Training (CT) and Continuous Deployment (CD) lifecycle. It actively monitors production data for concept drift, securely logs user telemetry, and orchestrates automated retraining pipelines with a strict "Human-in-the-Loop" (HITL) approval gate to guarantee model safety before production rollout.

## 2. Logical Architecture

The system is decoupled into five distinct functional layers, ensuring that infrastructure scaling, model training, and user inference do not block one another.

```text
+-----------------------------------------------------------------------------------+
|                              USER / PRESENTATION LAYER                            |
|  [ Web Frontend (Nginx) ] <--------> [ REST Clients / Downstream Microservices ]  |
+-----------------------------------------------------------------------------------+
                                  | (HTTP/JSON)
                                  v
+-----------------------------------------------------------------------------------+
|                                 SERVING & API LAYER                               |
|  [ FastAPI Gateway ]                                                              |
|   - Request Validation     - Traffic Routing     - PII Encryption                 |
|   - Telemetry Logging      - Local/Remote Model Toggling                          |
+-----------------------------------------------------------------------------------+
         | (Encrypted Telemetry)                   | (Inference Requests)
         v                                         v
+----------------------------------+    +-------------------------------------------+
|         STATE & DATA LAYER       |    |         MODEL REGISTRY & TRACKING         |
|  [ SQLite ] : Inference Logs     |    |  [ MLflow Server ] : Experiment Tracking  |
|  [ PostgreSQL ] : Orchestration  |    |  [ MLflow Model ] : Live Model Serving    |
|  [ DVC / Git ] : Data Versioning |    |  [ Artifact Store ] : Weights & Metrics   |
+----------------------------------+    +-------------------------------------------+
         | (Periodic Log Extraction)               ^ (Model Promotion / Tracking)
         v                                         |
+-----------------------------------------------------------------------------------+
|                                ORCHESTRATION LAYER                                |
|  [ Apache Airflow ]                                                               |
|   - DAG 1: Daily Drift Monitoring (PSI Calculation)                               |
|   - DAG 2: Continuous Training (Data Merge -> DVC Prep -> MLflow Train -> Alert)  |
+-----------------------------------------------------------------------------------+
                                  | (Metrics & Alerts)
                                  v
+-----------------------------------------------------------------------------------+
|                                OBSERVABILITY LAYER                                |
|  [ Prometheus ] : Time-series metrics scraping (API & System Health)              |
|  [ Grafana ] : Visual dashboards for Latency, Drift, and Fraud Rates              |
|  [ Alertmanager ] : SMTP routing for SLA breaches and human-review requests       |
+-----------------------------------------------------------------------------------+
```
## 3. Core Component Responsibilities
### 3.1. Serving & API Layer (FastAPI)
Acts as the single entry point for all predictions and user feedback. It abstracts the underlying model infrastructure from the frontend. It features a dynamic routing toggle, allowing the system to serve predictions from a fast, local memory cache (for development) or proxy requests to a dedicated MLflow serving container (for production scaling).

### 3.2. State & Data Layer (DVC & Databases)
Manages the immutable truth of the system. Raw CSV datasets and large sparse matrices are versioned via Data Version Control (DVC), while a PostgreSQL instance acts as the backend for Airflow and MLflow state management. A dedicated SQLite database handles high-throughput inference logging.

### 3.3. Orchestration Layer (Airflow)
The automation engine of the platform. It executes time-bound workflows (e.g., daily drift checks) and event-driven workflows (e.g., triggering a retraining cycle when new labeled data is ingested). Airflow bridges the gap between raw data engineering (DVC) and model experimentation (MLflow).

### 3.4. Model Registry & Tracking (MLflow)
Serves as the system of record for all machine learning experiments. It tracks hyperparameters, code versions, metrics (F1, AUC), and artifacts (confusion matrices). It also houses the Model Registry, managing the lifecycle stages of models from Staging to @champion (Production).

### 3.5. Observability Layer (Prometheus Stack)
Provides real-time visibility into system health and model behavior. It scrapes latency histograms and prediction distributions, surfacing them on Grafana dashboards. It acts as the system's immune response, immediately dispatching emails if SLAs are breached or if significant data drift occurs.

## 4. Key Data Flows
### 4.1. The Live Inference Flow
- A user submits a job posting via the Nginx frontend.

- FastAPI validates the schema and extracts preliminary features.

- FastAPI routes the numerical payload to the MLflow Model Serving container.

- The prediction is returned. FastAPI encrypts all Personally Identifiable Information (PII) using a Fernet cipher and commits the record to the SQLite database.

- The API responds to the user while asynchronously exposing latency and confidence metrics to Prometheus.

### 4.2. The Continuous Training (CT) Flow
- **Trigger:** An admin triggers the Retraining DAG in Airflow (often prompted by a Drift Alert).

- **Extraction:** Airflow safely extracts and decrypts user-validated feedback from SQLite.

- **Data Prep:** The new records are merged with the root training dataset, and Airflow delegates the heavy feature engineering (TF-IDF, Matrix Assembly) to the DVC pipeline.

- **Training:** Airflow invokes the training script, forcibly routing all metrics and the newly generated model over the HTTP network to the MLflow server.

- **Human Handoff:** Airflow halts, preventing automatic deployment. It sends an HTML email to the admin detailing the new model's performance, requiring human validation in the MLflow UI before the model is manually promoted to production.

## 5. Security & Compliance Posture
* **PII Protection:** Job descriptions, company profiles, and user feedback are symmetrically encrypted at rest inside the inference database, ensuring compliance with data privacy standards.

* **Deployment Safety:** The system explicitly prohibits automated Continuous Deployment (CD) of models, utilizing a Human-in-the-Loop registry promotion strategy to prevent poisoned data from silently degrading production accuracy.
