"""
DAG 2: Continuous Training Pipeline
Triggered manually or via Airflow UI/REST API.
Orchestrates the full retraining pipeline:
  1. Export decrypted, labeled inference logs from PostgreSQL
  2. Merge with root training data (fake_job_postings.csv)
  3. Validate merged dataset
  4. Run DVC pipeline (up to feature engineering)
  5. Trigger MLflow training run with the newly prepped data
  6. Send a beautiful HTML alert via EmailOperator for human review & commit
"""

from datetime import datetime, timedelta
import json
import os
import sys

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator

sys.path.insert(0, '/opt/airflow/src')

default_args = {
    'owner': 'da5402_project',
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
    'email_on_failure': True,
    'email_on_retry': False,
}

SOURCE_DATA_PATH = '/opt/airflow/data/raw/fake_job_postings.csv'
RETRAIN_DATA_PATH = '/opt/airflow/data/raw/fake_job_postings_retrain.csv'
PROJECT_DIR = '/opt/airflow'
API_HEALTH_URL = 'http://api:8000/ready'

# Data Prep & State Management
def export_labeled_inference_logs(**kwargs):
    import sqlite3
    import pandas as pd
    from cryptography.fernet import Fernet
    import subprocess 
    logger = kwargs['ti'].log

    fernet_key = os.getenv('AIRFLOW__CORE__FERNET_KEY')
    cipher = Fernet(fernet_key.encode()) if fernet_key else None

    def decrypt(text):
        if not text or pd.isna(text) or not cipher: return text
        try: return cipher.decrypt(str(text).encode()).decode()
        except: return text

    DB_PATH = '/opt/airflow/data/inference.db'
    conn = sqlite3.connect(DB_PATH)

    cur = conn.execute("""
        SELECT
            title, description, company_profile, requirements,
            employment_type, has_company_logo, has_questions,
            salary_range, prediction, user_label
        FROM inference_log
        WHERE user_label IS NOT NULL
    """)

    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    conn.close()

    df = pd.DataFrame(rows, columns=columns)
    
    if len(df) == 0:
        logger.warning("No labeled inference data found. Retraining will use original data only.")
        kwargs['ti'].xcom_push(key='has_new_data', value=False)
        return

    encrypted_cols = ['title', 'description', 'company_profile', 'requirements', 'salary_range', 'user_label']
    for col in encrypted_cols:
        if col in df.columns:
            df[col] = df[col].apply(decrypt)

    df['fraudulent'] = df['user_label'].apply(lambda x: 1 if x == 'Fraudulent' else 0)
    df = df.drop(columns=['prediction', 'user_label'])

    out_path = '/opt/airflow/data/raw/inference_labeled.csv'
    subprocess.run(f"rm -f {out_path}", shell=True, check=False)
    df.to_csv(out_path, index=False)
    kwargs['ti'].xcom_push(key='has_new_data', value=True)


def merge_datasets(**kwargs):
    import pandas as pd
    import subprocess
    logger = kwargs['ti'].log
    ti = kwargs['ti']

    has_new_data = ti.xcom_pull(key='has_new_data', task_ids='export_inference_logs')
    original_df = pd.read_csv(SOURCE_DATA_PATH)

    if has_new_data:
        inference_df = pd.read_csv('/opt/airflow/data/raw/inference_labeled.csv')
        merged_df = pd.concat([original_df, inference_df], ignore_index=True)
    else:
        merged_df = original_df

    merged_df = merged_df.drop_duplicates(subset=['title', 'description'])
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    subprocess.run(f"rm -f {RETRAIN_DATA_PATH}", shell=True, check=False)
    merged_df.to_csv(RETRAIN_DATA_PATH, index=False)


def validate_retrain_data(**kwargs):
    import pandas as pd
    df = pd.read_csv(RETRAIN_DATA_PATH)
    required_cols = ['title', 'description', 'fraudulent']
    missing = [c for c in required_cols if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    if len(df) < 1000: raise ValueError(f"Too few records: {len(df)}")


def backup_current_model(**kwargs):
    import shutil
    import os
    backup_dir = '/opt/airflow/data/production/backup'
    os.makedirs(backup_dir, exist_ok=True)
    files_to_backup = [
        '/opt/airflow/data/production/model.pkl',
        '/opt/airflow/data/production/tfidf_vectorizer.pkl',
        '/opt/airflow/data/production/metadata.json',
    ]
    for f in files_to_backup:
        if os.path.exists(f):
            shutil.copyfile(f, os.path.join(backup_dir, os.path.basename(f)))


def swap_source_data(**kwargs):
    import subprocess
    subprocess.run(f"cp {SOURCE_DATA_PATH} {SOURCE_DATA_PATH}.bak", shell=True, check=False)
    subprocess.run(f"rm -f {SOURCE_DATA_PATH}", shell=True, check=False)
    result = subprocess.run(f"cp {RETRAIN_DATA_PATH} {SOURCE_DATA_PATH}", shell=True)
    if result.returncode != 0: raise RuntimeError("Failed to swap retrain data.")


# Health & Email Alerts
def verify_api_health(**kwargs):
    import requests
    logger = kwargs['ti'].log
    try:
        response = requests.get(API_HEALTH_URL, timeout=10)
        if response.status_code == 200:
            logger.info("API is healthy. Production is stable.")
        else:
            raise ValueError(f"API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Cannot connect to API at {API_HEALTH_URL}")


def build_review_email(**kwargs):
    ti = kwargs['ti']
    email_body = f"""
    <html><body style="font-family:sans-serif; color:#333; max-width:600px; margin:0 auto;">
      <div style="background:#4da6ff; padding:24px; border-radius:8px 8px 0 0;">
        <h1 style="color:white; margin:0;">🚀 Human Review Required</h1>
        <p style="color:rgba(255,255,255,0.8); margin:8px 0 0;">JobGuard — Model Retraining Complete</p>
      </div>
      <div style="background:#f9f9f9; padding:24px; border-radius:0 0 8px 8px; border:1px solid #eee;">
        <p>A new model has been successfully trained and pushed to the MLflow Registry. It requires human validation.</p>
        
        <h3>Action Required:</h3>
        <ol style="line-height:1.6;">
            <li>Navigate to your <strong><a href="http://localhost:5000">MLflow UI</a></strong>.</li>
            <li>Review the newly generated metrics in the <code>jobguard-automated-retraining-exp</code> experiment.</li>
            <li>If the metrics meet production standards, assign it the <code>@champion</code> alias.</li>
            <li><strong>Finalize the Data Update:</strong> Open your host machine terminal and run:
                <div style="background:#2d2d2d; color:#fff; padding:10px; border-radius:5px; margin-top:5px; font-family:monospace;">
                    dvc push<br>
                    git add dvc.lock data/**/*.dvc<br>
                    git commit -m "Accept Airflow retrained data"<br>
                    git push
                </div>
            </li>
        </ol>
        <hr style="border:none; border-top:1px solid #eee; margin:20px 0;">
        <p style="color:#999; font-size:12px;">This is an automated handoff from the JobGuard MLOps pipeline.</p>
      </div>
    </body></html>
    """
    ti.xcom_push(key='review_email_body', value=email_body)

# DAG DEFINITION
with DAG(
    dag_id='fake_job_retraining_pipeline',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 4, 19),
    catchup=False,
    tags=['retraining', 'fake-job-detector', 'mlflow'],
    # Add this params dictionary to generate the UI form
    params={
        "n_estimators": Param(200, type="integer"),
        "max_depth": Param(5, type="integer"),
        "learning_rate": Param(0.1, type="number"),
        "subsample": Param(0.8, type="number"),
    }
) as dag:

    t1 = PythonOperator(task_id='export_inference_logs', python_callable=export_labeled_inference_logs)
    t2 = PythonOperator(task_id='merge_datasets', python_callable=merge_datasets)
    t3 = PythonOperator(task_id='validate_retrain_data', python_callable=validate_retrain_data)
    t4 = PythonOperator(task_id='backup_current_state', python_callable=backup_current_model)
    t5 = PythonOperator(task_id='swap_source_data', python_callable=swap_source_data)

    t6 = BashOperator(
        task_id='run_data_prep_and_mlflow_training',
        bash_command=f"""
        cd {PROJECT_DIR}
        export GIT_PYTHON_REFRESH=quiet
        export MLFLOW_TRACKING_URI="http://mlflow:5000"
        export MLFLOW_EXPERIMENT_NAME="jobguard-automated-retraining-exp"
        
        # Fix the Git "Dubious Ownership" error for mounted volumes!
        git config --global --add safe.directory {PROJECT_DIR}
        
        # Run DVC prep
        dvc repro -f split preprocess featurize_test 2>&1
        
        # Train with dynamically injected Airflow UI Parameters
        python src/model/train.py --use-mlflow \\
            --n_estimators {{{{ params.n_estimators }}}} \\
            --max_depth {{{{ params.max_depth }}}} \\
            --learning_rate {{{{ params.learning_rate }}}} \\
            --subsample {{{{ params.subsample }}}}
        """
    )

    t7 = PythonOperator(task_id='verify_api_health', python_callable=verify_api_health)
    t8_build = PythonOperator(task_id='build_review_email', python_callable=build_review_email)
    
    t8_send = EmailOperator(
        task_id='send_review_email',
        to='{{ var.value.get("alert_email") }}', 
        subject='[ACTION REQUIRED] JobGuard Model Awaiting Review — {{ ds }}',
        html_content="{{ ti.xcom_pull(task_ids='build_review_email', key='review_email_body') }}",
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8_build >> t8_send



    