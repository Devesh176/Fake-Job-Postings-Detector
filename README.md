# Fake-Job-Postings-Detector

This repo contains a end to end fake job posting detection pipeline using DVC, MLflow, Airflow, Alermanager, Prometheus and Grafana as backend. It uses FastAPI for frontend application.

## Dataset used
[Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

## Setup

1. Install Python dependencies (create venv recommended)
   
     ```bash
     python3 -m pip install -r requirements.txt
     ```

2. Create or update `.env` with MLflow/Airflow settings
   ```dotenv
   cp .env.example .env
   ```
   Modify the .env with necessary configuration and information
   ```dotenv
    AIRFLOW_SECRET_KEY=your_secret_key_here
    AIRFLOW_FERNET_KEY=your_secret_key_here
    # export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_TRACKING_URI=mlflow_tracking_uri
    export MLFLOW_ARTIFACT_ROOT=./mlflow_artifacts
    export MLFLOW_ARTIFACT_URI=mlflow_tracking_uri
    # MODEL_SERVE_MODE=mlflow
    MODEL_SERVE_MODE=local
    MLFLOW_EXPERIMENT_NAME=fake-job-detector-exp
    export MLFLOW_PYTHON_BIN=/usr/bin/python3
    MLFLOW_SERVE_URL=http://mlflow_model:5001/
    PSI_THRESHOLD=0.2

    SMTP_HOST=live.smtp.mailtrap.io
    SMTP_PORT=2525
    SMTP_USER=api
    SMTP_PASSWORD=your_mailtrap_api_key
    SMTP_FROM=hello@demomailtrap.co
    ALERT_TO_EMAIL=recipient@example.com
   ```

   **Note**: Generate secure keys and place them in `.env`:
  ```bash
  python3 - <<'PY'
  import secrets, base64, os
  print('AIRFLOW_SECRET_KEY=' + secrets.token_hex(32))
  print('AIRFLOW_FERNET_KEY=' + base64.urlsafe_b64encode(os.urandom(32)).decode())
  PY
  ```

3. Load environment variables
   ```bash
   source .env
   ```

4. Setup data folder
   ```bash
    unzip dvc-remote.zip
   ```
   change the path inside .dvc/config
   ```bash
   nano .dvc/config
   ```
   Change the url to the unzipped path
   ```bash
   ['remote "local_remote"']
    url = path_to_unzipped_directory
    ```
   The fetch the tracked models and data
   ```bash
   dvc pull
   dvc status
   ```
5. Run dvc pipeline (Optional)
   ```bash
    dvc repro --force
   ```
## Start Docker Services
Current docker service are-
- postgress 
- mlflow (port 5000)
- api (port 8000)
- airflow (port 8080)
- alertmanager (port 9093)
- prometheus (port 9090)
- grafana (port 3001)

To start the docker services run:
```bash
docker compose up -d --build
# If already built run: docker compose up -d
```
Veify the container status with
```bash
docker ps
```
**Note**:  It can be verified by visiting the links on browser as well

## MLflow commands

Run the MLflow entry points from the repo root.
Use `--env-manager=local` if you do not want to rely on Conda, and use `--no-conda` if Conda is not installed.

Train:
```bash
export MLFLOW_PYTHON_BIN=/usr/bin/python3
mlflow run . -e train --env-manager=local --no-conda --experiment-name fake-job-detector-exp --run-name my_exp \
  -P n_estimators=200 -P max_depth=5 -P learning_rate=0.1 -P subsample=0.8
```

Evaluate:
```bash
mlflow run . -e evaluate --env-manager=local --no-conda --experiment-name fake-job-detector-exp
```

Register:
```bash
mlflow run . -e register --env-manager=local --no-conda --experiment-name fake-job-detector-exp
```

Or run the full sequence:
```bash
mlflow run . -e train --env-manager=local --no-conda --experiment-name fake-job-detector-exp --run-name my_exp \
  -P n_estimators=200 -P max_depth=5 -P learning_rate=0.1 -P subsample=0.8 && \
mlflow run . -e evaluate --env-manager=local --no-conda --experiment-name fake-job-detector-exp && \
mlflow run . -e register --env-manager=local --no-conda --experiment-name fake-job-detector-exp
```
To export mlflow runs data:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=fake-job-detector-exp

python3 src/export_mlflow_runs.py
```

## Airflow

Add an Airflow admin user manually if needed:
```bash
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

## API

The project includes a FastAPI-based web API for real-time fake job posting detection.

### Running the API

Start the API with Docker Compose:
```bash
docker compose up --build api
```

The API will be available at `http://localhost:8000`.

### API Endpoints

- `GET /` - Web UI for job posting classification
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /mode` - Current model serving mode (local or MLflow)
- `POST /predict` - Classify a job posting (JSON payload with job details)
- `POST /feedback` - Submit user feedback on a prediction
- `GET /history` - Recent prediction history
- `GET /stats` - Prediction statistics
- `GET /metrics` - Prometheus metrics

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Software Engineer",
    "description": "We are looking for a skilled software engineer...",
    "company_profile": "Tech company description",
    "requirements": "Bachelor degree, 3+ years experience",
    "employment_type": "Full-time",
    "has_company_logo": 1,
    "has_questions": 0,
    "salary_range": "$80k-$100k",
    "telecommuting": 0
  }'
```

### Database

Predictions are logged to a local SQLite database at `data/inference.db` on the host machine.


## Running Tests
```bash
# Install test dependencies
pip install pytest pytest-html

# Run all tests with verbose output
pytest tests/test_suite.py -v --tb=short

# Generate markdown report
python tests/run_tests.py

# Run only acceptance criteria tests
pytest tests/test_suite.py::TestModelPerformance -v
```

## Notes

- `mlflow` entry points are defined in `MLproject`.
- Use `--no-conda` when Conda is unavailable or when running in the current Python environment.
- Do not track generated `mlflow_artifacts/` with Git or DVC unless you explicitly want to version local MLflow artifact caches.
