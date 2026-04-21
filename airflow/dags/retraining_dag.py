"""
DAG 2: Manual Retraining Pipeline
Triggered manually by admin via Airflow UI or REST API.
Orchestrates the full retraining pipeline:
  1. Export decrypted, labeled inference logs from PostgreSQL
  2. Merge with root training data (fake_job_postings.csv)
  3. Validate merged dataset
  4. Run DVC pipeline (dvc repro --force) - ensures proper train/test split
  5. Evaluate new model
  6. Compare with current production model
  7. Promote if better — run dvc push
  8. Verify new model is live via API health check
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/opt/airflow/src')

# Default Args 
default_args = {
    'owner': 'da5402_project',
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
    'email_on_failure': False,
    'email_on_retry': False,
}

# Updated Paths to target the root data file before the DVC split stage
SOURCE_DATA_PATH = '/opt/airflow/data/raw/fake_job_postings.csv'
RETRAIN_DATA_PATH = '/opt/airflow/data/raw/fake_job_postings_retrain.csv'

METRICS_PATH = '/opt/airflow/data/processed/metrics.json'
PRODUCTION_METADATA_PATH = '/opt/airflow/data/production/metadata.json'
PROJECT_DIR = '/opt/airflow'
API_HEALTH_URL = 'http://api:8000/ready'

# Task Functions 

def export_labeled_inference_logs(**kwargs):
    """
    Export inference logs that have user-confirmed labels.
    Decrypts the sensitive fields before merging with training data.
    """
    import sqlite3
    import pandas as pd
    from cryptography.fernet import Fernet
    import subprocess 
    logger = kwargs['ti'].log

    logger.info("Exporting and decrypting labeled inference logs from SQLite...")

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
        kwargs['ti'].xcom_push(key='labeled_count', value=0)
        kwargs['ti'].xcom_push(key='has_new_data', value=False)
        return

    encrypted_cols = ['title', 'description', 'company_profile', 'requirements', 'salary_range', 'user_label']
    for col in encrypted_cols:
        if col in df.columns:
            df[col] = df[col].apply(decrypt)


    df['fraudulent'] = df['user_label'].apply(lambda x: 1 if x == 'Fraudulent' else 0)
    df = df.drop(columns=['prediction', 'user_label'])

    labeled_count = len(df)
    logger.info(f"Exported and decrypted {labeled_count} labeled records.")
    
    out_path = '/opt/airflow/data/raw/inference_labeled.csv'
    
    # --- NEW FIX: Force delete the old file if it's locked by root ---
    logger.info(f"Cleaning up any old locked files at {out_path}...")
    subprocess.run(f"rm -f {out_path}", shell=True, check=False)
    # -----------------------------------------------------------------

    df.to_csv(out_path, index=False)
    kwargs['ti'].xcom_push(key='labeled_count', value=labeled_count)
    kwargs['ti'].xcom_push(key='has_new_data', value=True)


def merge_datasets(**kwargs):
    """
    Merge original root data with labeled inference logs.
    Deduplicates and shuffles before saving.
    """
    import pandas as pd
    import subprocess
    logger = kwargs['ti'].log
    ti = kwargs['ti']

    has_new_data = ti.xcom_pull(key='has_new_data', task_ids='export_inference_logs')
    original_df = pd.read_csv(SOURCE_DATA_PATH)
    logger.info(f"Original root data: {len(original_df)} records")

    if has_new_data:
        inference_df = pd.read_csv('/opt/airflow/data/raw/inference_labeled.csv')
        logger.info(f"New labeled inference data: {len(inference_df)} records")
        merged_df = pd.concat([original_df, inference_df], ignore_index=True)
    else:
        logger.info("No new data — using original root data only.")
        merged_df = original_df

    before = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['title', 'description'])
    after = len(merged_df)
    logger.info(f"Deduplicated: {before} → {after} records")

    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Cleaning up any old locked files at {RETRAIN_DATA_PATH}...")
    subprocess.run(f"rm -f {RETRAIN_DATA_PATH}", shell=True, check=False)
  
    merged_df.to_csv(RETRAIN_DATA_PATH, index=False)
    
    logger.info(f"Merged dataset saved to {RETRAIN_DATA_PATH} ({len(merged_df)} records)")
    ti.xcom_push(key='merged_count', value=len(merged_df))

def validate_retrain_data(**kwargs):
    """Validate the merged dataset before retraining."""
    import pandas as pd
    logger = kwargs['ti'].log

    df = pd.read_csv(RETRAIN_DATA_PATH)
    logger.info(f"Validating retrain dataset: {df.shape}")

    required_cols = ['title', 'description', 'fraudulent']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < 1000:
        raise ValueError(f"Too few records: {len(df)}")

    logger.info("Validation PASSED — proceeding with retraining swap.")


def backup_current_model(**kwargs):
    """Backup current production model before retraining."""
    import shutil
    import os
    logger = kwargs['ti'].log

    backup_dir = '/opt/airflow/data/production/backup'
    os.makedirs(backup_dir, exist_ok=True)

    files_to_backup = [
        '/opt/airflow/data/production/model.pkl',
        '/opt/airflow/data/production/tfidf_vectorizer.pkl',
        '/opt/airflow/data/production/metadata.json',
    ]

    for f in files_to_backup:
        if os.path.exists(f):
            dest = os.path.join(backup_dir, os.path.basename(f))
            # Use copyfile to copy contents only, bypassing permission metadata cloning
            shutil.copyfile(f, dest)
            logger.info(f"Backed up: {f} → {dest}")


def swap_source_data(**kwargs):
    """
    Swap fake_job_postings.csv with retrain data.
    Uses subprocess to bypass Python's strict permission handling on DVC's read-only files.
    """
    import subprocess
    logger = kwargs['ti'].log

    logger.info("Bypassing DVC read-only locks to swap root training data...")

    subprocess.run(f"cp {SOURCE_DATA_PATH} {SOURCE_DATA_PATH}.bak", shell=True, check=False)
    subprocess.run(f"rm -f {SOURCE_DATA_PATH}", shell=True, check=False)
    
    result = subprocess.run(f"cp {RETRAIN_DATA_PATH} {SOURCE_DATA_PATH}", shell=True)
    
    if result.returncode == 0:
        logger.info(f"Successfully swapped fake_job_postings.csv with retrain data")
    else:
        raise RuntimeError("Failed to copy retrain data into fake_job_postings.csv")


def evaluate_new_model(**kwargs):
    import json
    logger = kwargs['ti'].log
    ti = kwargs['ti']

    with open(METRICS_PATH, 'r') as f:
        new_metrics = json.load(f)

    try:
        with open(PRODUCTION_METADATA_PATH, 'r') as f:
            current_metadata = json.load(f)
        current_metrics = current_metadata.get('metrics', {})
        current_f1 = current_metrics.get('val_f1_fraud', 0.0)
        current_auc = current_metrics.get('val_roc_auc', 0.0)
    except FileNotFoundError:
        current_f1 = 0.0
        current_auc = 0.0

    new_f1 = new_metrics.get('val_f1_fraud', 0.0)
    new_auc = new_metrics.get('val_roc_auc', 0.0)

    meets_min_quality = new_auc >= 0.85
    not_worse_than_current = new_f1 >= (current_f1 - 0.02)
    should_promote = meets_min_quality and not_worse_than_current

    comparison = {
        'new_f1': new_f1, 'new_auc': new_auc,
        'current_f1': current_f1, 'current_auc': current_auc,
        'should_promote': should_promote,
    }

    logger.info(f"New F1: {new_f1:.4f} | Current F1: {current_f1:.4f}")
    logger.info(f"Should promote: {should_promote}")

    ti.xcom_push(key='should_promote', value=should_promote)
    ti.xcom_push(key='comparison', value=json.dumps(comparison))


def branch_on_evaluation(**kwargs):
    should_promote = kwargs['ti'].xcom_pull(key='should_promote', task_ids='evaluate_new_model')
    return 'promote_model' if should_promote else 'rollback_model'


def promote_model(**kwargs):
    import json
    logger = kwargs['ti'].log
    ti = kwargs['ti']
    comparison = json.loads(ti.xcom_pull(key='comparison', task_ids='evaluate_new_model'))

    metadata = {
        'model_path': 'data/production/model.pkl',
        'vectorizer_path': 'data/production/tfidf_vectorizer.pkl',
        'promoted_at': datetime.utcnow().isoformat(),
        'metrics': {'val_f1_fraud': comparison['new_f1'], 'val_roc_auc': comparison['new_auc']},
        'previous_metrics': {'val_f1_fraud': comparison['current_f1'], 'val_roc_auc': comparison['current_auc']},
        'trigger': kwargs.get('dag_run').conf.get('trigger_reason', 'manual'),
    }

    with open(PRODUCTION_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model promoted to production.")


def rollback_model(**kwargs):
    import subprocess
    import shutil
    import os
    logger = kwargs['ti'].log

    backup_dir = '/opt/airflow/data/production/backup'
    for fname in ['model.pkl', 'tfidf_vectorizer.pkl', 'metadata.json']:
        src = os.path.join(backup_dir, fname)
        dst = os.path.join('/opt/airflow/data/production', fname)
        if os.path.exists(src):
            # Use copyfile to copy contents only
            shutil.copyfile(src, dst)

    if os.path.exists(SOURCE_DATA_PATH + '.bak'):
        subprocess.run(f"rm -f {SOURCE_DATA_PATH}", shell=True)
        subprocess.run(f"cp {SOURCE_DATA_PATH}.bak {SOURCE_DATA_PATH}", shell=True)
        logger.info("Restored original fake_job_postings.csv from backup.")

    logger.warning("Production model rolled back.")


def verify_api_health(**kwargs):
    import requests
    logger = kwargs['ti'].log
    try:
        response = requests.get(API_HEALTH_URL, timeout=10)
        if response.status_code == 200:
            logger.info("API is healthy.")
        else:
            raise ValueError(f"API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Cannot connect to API at {API_HEALTH_URL}")


def log_retraining_summary(**kwargs):
    import json
    logger = kwargs['ti'].log
    ti = kwargs['ti']

    comparison_json = ti.xcom_pull(key='comparison', task_ids='evaluate_new_model')
    comparison = json.loads(comparison_json) if comparison_json else {}
    
    logger.info("=" * 70)
    logger.info("RETRAINING PIPELINE SUMMARY")
    logger.info(f"  Promoted         : {comparison.get('should_promote', False)}")
    logger.info("=" * 70)


# DAG Definition 
with DAG(
    dag_id='fake_job_retraining_pipeline',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 4, 19),
    catchup=False,
    tags=['retraining', 'fake-job-detector'],
) as dag:

    t1 = PythonOperator(task_id='export_inference_logs', python_callable=export_labeled_inference_logs)
    t2 = PythonOperator(task_id='merge_datasets', python_callable=merge_datasets)
    t3 = PythonOperator(task_id='validate_retrain_data', python_callable=validate_retrain_data)
    t4 = PythonOperator(task_id='backup_current_model', python_callable=backup_current_model)
    t5 = PythonOperator(task_id='swap_source_data', python_callable=swap_source_data)

    t6 = BashOperator(
        task_id='run_dvc_pipeline',
        bash_command=f'cd {PROJECT_DIR} && dvc repro --force 2>&1',
    )

    t7 = PythonOperator(task_id='evaluate_new_model', python_callable=evaluate_new_model)
    t8 = BranchPythonOperator(task_id='branch_on_evaluation', python_callable=branch_on_evaluation)
    
    t9_promote = PythonOperator(task_id='promote_model', python_callable=promote_model)
    t9_rollback = PythonOperator(task_id='rollback_model', python_callable=rollback_model)

    t10_push = BashOperator(
        task_id='dvc_push',
        bash_command=f'cd {PROJECT_DIR} && dvc push -r airflow_remote 2>&1',
        trigger_rule='all_success',
    )
    
    t11 = PythonOperator(task_id='verify_api_health', python_callable=verify_api_health)
    t12 = PythonOperator(task_id='log_retraining_summary', python_callable=log_retraining_summary, trigger_rule='all_done')

    # Pipeline Flow
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8
    t8 >> [t9_promote, t9_rollback]
    t9_promote >> t10_push >> t11 >> t12
    t9_rollback >> t12