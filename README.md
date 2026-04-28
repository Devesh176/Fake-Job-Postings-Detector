# Fake Job Postings Detector

This is a complete, production-ready MLOps platform built to detect fraudulent job postings in real-time. 

Under the hood, this system manages the entire ML lifecycle: from strict data versioning (DVC) and experiment tracking (MLflow), to real-time model serving (FastAPI), system observability (Prometheus/Grafana), and automated continuous training pipelines (Airflow) complete with Human-in-the-Loop validation.

## The Architecture Stack
* **Serving Layer:** FastAPI backend + Nginx frontend.
* **Data Versioning:** DVC (Data Version Control).
* **Experiment Tracking & Registry:** MLflow.
* **Orchestration:** Apache Airflow.
* **Observability:** Prometheus, Grafana, and Alertmanager.
* **Infrastructure:** Fully containerized via Docker Compose.

### Project Structure:
```bash
.
├── airflow
│   └── dags
│       ├── drift_monitoring_dag.py   # Daily drift monitoring dag
│       └── retraining_dag.py         # Retraining dag (manually triggered)
├── data                              # data directory (handled by DVC)
│   ├── baselines
│   │   └── training_baseline.json    # Training baseline     
│   ├── inference.db                  # Backend database store
│   ├── processed                     # Processed data
│   │   ├── model.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── train_processed.csv
│   │   ├── X_test.npz
│   │   ├── X_train.npz
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   ├── production                    # Models used by api backend
│   │   ├── model.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── raw                           # Raw data
│       ├── fake_job_postings.csv
│       ├── fake_job_postings.csv.dvc
│       ├── test.csv
│       └── train.csv
├── docker-compose.yml                # docker containers definitions
├── docs                              # documentation
│   ├── assets
│   │   └── style.css
│   ├── HLD.md                        # High Level Design Doc
│   ├── LLD.md                        # Low Level Desing Doc
│   └── report.html                   # Pytest report
├── dvc.lock  
├── dvc-remote.zip                    # zipped local DVC remote
├── dvc.yaml                          # DVC pipeline definition
├── eda.ipynb
├── infra                             # Storing Dockerfile Definitions
│   ├── Dockerfile                    # used by api
│   ├── Dockerfile.airflow            # Airflow image
│   ├── Dockerfile.mlflow             # MLFlow image
│   └── init-db.sql
├── LICENSE
├── mlflow_artifacts                  # Not included in repo but automatically created to store mlflow artifacts          
├── mlflow_runs.csv                   # MLFlow runs exported in csv
├── MLproject                       
├── monitoring                        
│   ├── alertmanager.yml              
│   ├── alert_rules.yml               # Alert rules
│   ├── grafana
│   │   ├── dashboards                # Grafana Dashboards
│   │   │   ├── api_health.json
│   │   │   ├── dashboards.yml
│   │   │   ├── drift_monitoring.json
│   │   │   └── model_performance.json
│   │   └── datasources
│   │       └── datasource.yml
│   └── prometheus.yml
├── pictures                          # Pictures of running containers
├── python_env.yaml                   # Python Environment file for MLproject
├── README.md
├── requirements-airflow.txt          # Requirements files
├── requirements.txt
├── src
│   ├── export_mlflow_runs.py         # Helper to extract mlflow runs as CSV
│   ├── main.py                       # Main API backend
│   ├── model                         
│   │   ├── evaluate.py               # Evaluation helper  
│   │   ├── register_dvc.py           # Model Promotion to production helper script
│   │   ├── register.py               
│   │   ├── train_dvc.py                  
│   │   └── train.py                  # Model training script
│   ├── pipeline                      
│   │   ├── eda_stats.py              # Storing baselines
│   │   ├── featurize_test.py         # feature creations
│   │   ├── __init__.py
│   │   ├── preprocess.py             # Data preprocessing
│   │   ├── split_data.py             
│   │   └── validate.py               # Model validation helper script
│   └── static
│       └── index.html                # API frontend
└── tests
    ├── conftest.py                   # Testing scripts
    ├── run_tests.py
    ├── test_plan.md
    └── test_suite.py
```

---

## Quickstart Setup

### 1. Environment Preparation
Clone the repository and set up your Python environment (Python 3.11+ recommended):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Configure Environment Variables
You need a .env file at the root of the project to securely pass keys to Docker, Airflow, and the API.

```bash
cd ~/Fake-Job-Postings-Detector
cp .env.example .env
```
The .env file containes follwing variables-
```bash
AIRFLOW_UID={id -u} # run command to get id
AIRFLOW_SECRET_KEY=your_secret_key_here
AIRFLOW_FERNET_KEY=your_secret_key_here
# export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_ARTIFACT_ROOT=./mlflow_artifacts
export MLFLOW_ARTIFACT_URI=http://localhost:5000
# MODEL_SERVE_MODE=mlflow
MODEL_SERVE_MODE=local
MLFLOW_EXPERIMENT_NAME=fake-job-detector-exp
export MLFLOW_PYTHON_BIN=/usr/bin/python3
MLFLOW_SERVE_URL=http://mlflow_model:5001/invocations
PSI_THRESHOLD=0.2

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=hello@example.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=hello@example.com # email corresponding to the above password 
ALERT_EMAIL=me22b176@smail.iitm.ac.in # replace with your email
```
To generate AIRFLOW secret keys run the follwing python commands-
```python
import secrets, base64, os
print(f"AIRFLOW_SECRET_KEY={secrets.token_hex(32)}")
print(f"AIRFLOW_FERNET_KEY={base64.urlsafe_b64encode(os.urandom(32)).decode()}")
```

### 3. Initialize DVC (Data Version Control)
The raw data and heavy model artifacts are tracked by DVC. To link your local repository to the data:
```bash
# 0. pull the dvc-remote.zip
git lfs pull

# 1. Unzip the local remote directory
unzip dvc-remote.zip

# 2. Update your DVC config to point to the absolute path of that unzipped folder
nano .dvc/config 
# Change the url line to: url = /your/absolute/path/to/dvc-remote

# 3. Pull the tracked data
dvc pull
```
---

## Running the Platform
Once your .env and DVC are set up, spin up the entire MLOps stack using Docker Compose:

```bash
docker compose up -d --build
```
You can verify everything is running smoothly by running docker ps and accessing the following local ports:

* Frontend UI: http://localhost:80

* FastAPI Backend: http://localhost:8000/docs

* MLflow Tracking Server: http://localhost:5000

* Airflow UI: http://localhost:8080 (Login: admin / admin)

* Grafana Dashboards: http://localhost:3001 (Login: admin / admin)

* Prometheus Targets: http://localhost:9090

---

## Training & Experimentation
This project utilizes a Y-Shaped Training Architecture. Our core `src/model/train.py` script acts differently depending on how you invoke it.

### Option A: Standard DVC Pipeline (Local Execution)
If you are iterating locally and just want to update your DVC artifacts (model.pkl and metrics.json), run:

```bash
dvc repro
```

### Option B: MLflow Experiment Tracking
1. If you want to track hyperparameters, log confusion matrices, and push a model to the registry, export your tracking URI and pass the `--use-mlflow` flag:

  ```bash
  export MLFLOW_TRACKING_URI="http://localhost:5000"
  python src/model/train.py --use-mlflow --n_estimators 300 --max_depth 6
  ```

2. You can also use the standard mlflow run . command utilizing the MLproject file if you prefer CLI execution.
Run the MLflow entry points from the repo root.
Use `--env-manager=local` if you do not want to rely on Conda, and use `` if Conda is not installed.
- **Train:**
  ```bash
  export MLFLOW_PYTHON_BIN=/usr/bin/python3
  export MLFLOW_TRACKING_URI="http://localhost:5000"
  mlflow run . -e train --env-manager=local --experiment-name fake-job-detector-exp --run-name my_exp \
  -P n_estimators=200 -P max_depth=5 -P learning_rate=0.1 -P subsample=0.8
  ```

- **Evaluate:**
  ```bash
  export MLFLOW_TRACKING_URI="http://localhost:5000"
  mlflow run . -e evaluate --env-manager=local  --experiment-name fake-job-detector-exp
  ```

- **Register:**
  ```bash
  export MLFLOW_TRACKING_URI="http://localhost:5000"
  mlflow run . -e register --env-manager=local  --experiment-name fake-job-detector-exp
  ```

- **Or run the full sequence:**
  ```bash
  export MLFLOW_TRACKING_URI="http://localhost:5000"
  mlflow run . -e train --env-manager=local  --experiment-name fake-job-detector-exp --run-name my_exp \
  -P n_estimators=200 -P max_depth=5 -P learning_rate=0.1 -P subsample=0.8 && \
  mlflow run . -e evaluate --env-manager=local  --experiment-name fake-job-detector-exp && \
  mlflow run . -e register --env-manager=local  --experiment-name fake-job-detector-exp
  ```

**Note:** To export mlflow runs data as csv:
  ```bash
  export MLFLOW_TRACKING_URI=http://localhost:5000
  export MLFLOW_EXPERIMENT_NAME=fake-job-detector-exp # or change to other experiment name
  python3 src/export_mlflow_runs.py
  ```

---

## Automated MLOps (Airflow)
We have two core DAGs operating in Airflow to automate model degradation detection and recovery.

1. **Daily Drift Monitoring (fake_job_drift_monitoring):**
  Runs daily to extract the last 24 hours of inference logs from SQLite. It calculates Population Stability Index (PSI) against our baseline data.
    - *Healthy:* Sends a daily heartbeat summary email.
    - *Drift Detected:* Fires a critical alert to your inbox detailing the drifted features and recommending a retraining cycle.

2. **Continuous Training Pipeline (fake_job_retraining_pipeline):**
  When triggered, this DAG executes our Human-in-the-Loop retraining workflow:
    - Extracts and decrypts new user-labeled data from the API.
    - Merges it with the root dataset.
    - Automatically runs DVC feature engineering (split, preprocess, featurize).
    - Trains a new model via MLflow.
    - Halts and alerts an admin via email for manual review in the MLflow UI. If the F1/AUC scores look good, the admin applies the @champion alias and pushes the DVC updates to Git.

---

## API Usage
The FastAPI backend securely handles inference and encrypts PII before logging.

Example Prediction Request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior Machine Learning Engineer",
    "description": "We are looking for a skilled MLOps engineer...",
    "company_profile": "A fast-growing tech startup...",
    "requirements": "Bachelor degree, 3+ years experience with Airflow and MLflow.",
    "employment_type": "Full-time",
    "has_company_logo": 1,
    "has_questions": 1,
    "salary_range": "$120k-$150k",
    "user_label": "Fraudulent" 
  }'
  # Note: user_label: is optional and can take two values "Fradulent" or "Legitimate"
```
---

## Testing
To run the automated test suite locally:
```bash
# Install test dependencies
pip install pytest pytest-html

# Run the full test suite
pytest tests/test_suite.py -v --tb=short
```
## Important Notes
**Data Privacy:** All PII hitting the inference_log database is symmetrically encrypted using the `AIRFLOW__CORE__FERNET_KEY`. If you lose this key, your historic inference data cannot be decrypted for retraining.

**Artifact Caches:** Do not track the dynamically generated mlflow_artifacts/ directory with Git or DVC. It is automatically added to .gitignore.