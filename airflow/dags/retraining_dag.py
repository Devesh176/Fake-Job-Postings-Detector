"""
DAG 2: Manual Retraining Pipeline
Triggered manually by admin via Airflow UI or REST API.
Orchestrates the full retraining pipeline:
  1. Export labeled inference logs from PostgreSQL
  2. Merge with original training data
  3. Validate merged dataset
  4. Run DVC pipeline (dvc repro --force)
  5. Evaluate new model
  6. Compare with current production model
  7. Promote if better — run dvc push
  8. Verify new model is live via API health check
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
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

TRAIN_DATA_PATH = '/opt/airflow/data/raw/train.csv'
RETRAIN_DATA_PATH = '/opt/airflow/data/raw/train_retrain.csv'
METRICS_PATH = '/opt/airflow/data/processed/metrics.json'
PRODUCTION_METADATA_PATH = '/opt/airflow/data/production/metadata.json'
PROJECT_DIR = '/opt/airflow'
# API_HEALTH_URL = 'http://api:8000/ready'
API_HEALTH_URL = 'http://api:8000/ready'

import stat
os.chmod(RETRAIN_DATA_PATH, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

# Task Functions 

def export_labeled_inference_logs(**kwargs):
    """
    Export inference logs that have user-confirmed labels (feedback).
    These are the records where users marked the prediction as correct/incorrect.
    Only exports records with feedback — unlabeled records are not used for retraining.
    """
    import sqlite3
    import pandas as pd
    logger = kwargs['ti'].log

    logger.info("Exporting labeled inference logs from SQLite...")

    DB_PATH = '/opt/airflow/data/inference.db'
    conn = sqlite3.connect(DB_PATH)

    cur = conn.execute("""
        SELECT
            title,
            description,
            company_profile,
            requirements,
            employment_type,
            has_company_logo,
            has_questions,
            salary_range,
            CASE
                WHEN user_feedback = 'correct' THEN
                    CASE WHEN prediction = 'Fraudulent' THEN 1 ELSE 0 END
                WHEN user_feedback = 'incorrect' THEN
                    CASE WHEN prediction = 'Fraudulent' THEN 0 ELSE 1 END
                ELSE NULL
            END as fraudulent
        FROM inference_log
        WHERE user_feedback IS NOT NULL
    """)

    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    conn.close()

    df = pd.DataFrame(rows, columns=columns)
    df = df.dropna(subset=['fraudulent'])
    labeled_count = len(df)

    logger.info(f"Exported {labeled_count} labeled records from inference logs.")
    kwargs['ti'].xcom_push(key='labeled_count', value=labeled_count)

    if labeled_count == 0:
        logger.warning("No labeled inference data found. Retraining will use original data only.")
        kwargs['ti'].xcom_push(key='has_new_data', value=False)
        return

    df.to_csv('/opt/airflow/data/raw/inference_labeled.csv', index=False)
    kwargs['ti'].xcom_push(key='has_new_data', value=True)
    logger.info(f"Saved labeled data to /opt/airflow/data/raw/inference_labeled.csv")


def merge_datasets(**kwargs):
    """
    Merge original training data with labeled inference logs.
    Deduplicates and shuffles before saving as new training data.
    """
    import pandas as pd
    logger = kwargs['ti'].log
    ti = kwargs['ti']

    has_new_data = ti.xcom_pull(key='has_new_data', task_ids='export_inference_logs')
    original_df = pd.read_csv(TRAIN_DATA_PATH)
    logger.info(f"Original training data: {len(original_df)} records")

    if has_new_data:
        inference_df = pd.read_csv('/opt/airflow/data/raw/inference_labeled.csv')
        logger.info(f"New labeled inference data: {len(inference_df)} records")
        merged_df = pd.concat([original_df, inference_df], ignore_index=True)
    else:
        logger.info("No new data — using original training data only.")
        merged_df = original_df

    # Deduplicate on title + description
    before = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['title', 'description'])
    after = len(merged_df)
    logger.info(f"Deduplicated: {before} → {after} records")

    # Shuffle
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    merged_df.to_csv(RETRAIN_DATA_PATH, index=False)
    logger.info(f"Merged dataset saved to {RETRAIN_DATA_PATH} ({len(merged_df)} records)")
    logger.info(f"Fraud rate: {merged_df['fraudulent'].mean():.4f}")

    ti.xcom_push(key='merged_count', value=len(merged_df))
    ti.xcom_push(key='fraud_rate', value=float(merged_df['fraudulent'].mean()))


def validate_retrain_data(**kwargs):
    """
    Validate the merged dataset before retraining.
    Checks schema, missing values, class distribution.
    Fails the task if data is invalid.
    """
    import pandas as pd
    logger = kwargs['ti'].log

    df = pd.read_csv(RETRAIN_DATA_PATH)
    logger.info(f"Validating retrain dataset: {df.shape}")

    # Check 1: Required columns
    required_cols = ['title', 'description', 'fraudulent']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info(" Required columns present")

    # Check 2: Target variable is binary
    unique_vals = set(df['fraudulent'].dropna().unique())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target variable has unexpected values: {unique_vals}")
    logger.info(" Target variable is binary")

    # Check 3: Minimum records
    if len(df) < 1000:
        raise ValueError(f"Too few records for retraining: {len(df)}. Minimum 1000 required.")
    logger.info(f" Sufficient records: {len(df)}")

    # Check 4: Class imbalance not too extreme
    fraud_rate = df['fraudulent'].mean()
    if fraud_rate < 0.01:
        raise ValueError(f"Fraud rate too low: {fraud_rate:.4f}. Check data quality.")
    logger.info(f" Fraud rate acceptable: {fraud_rate:.4f}")

    # Check 5: No excessive missing values
    missing_pct = df['description'].isnull().mean()
    if missing_pct > 0.5:
        raise ValueError(f"Too many missing descriptions: {missing_pct:.1%}")
    logger.info(" Missing values within acceptable range")

    logger.info("Validation PASSED — proceeding with retraining.")


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
            shutil.copy(f, dest)
            logger.info(f"Backed up: {f} → {dest}")
        else:
            logger.warning(f"File not found for backup: {f}")

    logger.info("Current model backed up successfully.")


def swap_train_data(**kwargs):
    """
    Swap train.csv with retrain.csv so DVC pipeline uses the merged dataset.
    Saves original train.csv as a backup.
    """
    import shutil
    logger = kwargs['ti'].log

    # Backup original — content only, no permission copy
    shutil.copyfile(TRAIN_DATA_PATH, TRAIN_DATA_PATH + '.bak')
    logger.info(f"Backed up original train.csv → {TRAIN_DATA_PATH}.bak")

    # Swap — content only
    shutil.copyfile(RETRAIN_DATA_PATH, TRAIN_DATA_PATH)
    logger.info(f"Swapped train.csv with retrain data ({RETRAIN_DATA_PATH})")


def evaluate_new_model(**kwargs):
    """
    Compare new model metrics against current production model.
    Pushes comparison result to XCom.
    """
    import json
    logger = kwargs['ti'].log
    ti = kwargs['ti']

    # Load new model metrics
    with open(METRICS_PATH, 'r') as f:
        new_metrics = json.load(f)

    logger.info(f"New model metrics: {json.dumps(new_metrics, indent=2)}")

    # Load current production metrics for comparison
    try:
        with open(PRODUCTION_METADATA_PATH, 'r') as f:
            current_metadata = json.load(f)
        current_metrics = current_metadata.get('metrics', {})
        current_f1 = current_metrics.get('val_f1_fraud', 0.0)
        current_auc = current_metrics.get('val_roc_auc', 0.0)
        logger.info(f"Current production model — F1: {current_f1:.4f}, AUC: {current_auc:.4f}")
    except FileNotFoundError:
        logger.warning("No current production model found. New model will be promoted by default.")
        current_f1 = 0.0
        current_auc = 0.0

    new_f1 = new_metrics.get('val_f1_fraud', 0.0)
    new_auc = new_metrics.get('val_roc_auc', 0.0)

    # Promotion criteria:
    # 1. New model must have AUC >= 0.85
    # 2. New model must not be worse than current by more than 2%
    meets_min_quality = new_auc >= 0.85
    not_worse_than_current = new_f1 >= (current_f1 - 0.02)
    should_promote = meets_min_quality and not_worse_than_current

    comparison = {
        'new_f1': new_f1,
        'new_auc': new_auc,
        'current_f1': current_f1,
        'current_auc': current_auc,
        'meets_min_quality': meets_min_quality,
        'not_worse_than_current': not_worse_than_current,
        'should_promote': should_promote,
    }

    logger.info(f"New F1: {new_f1:.4f} | Current F1: {current_f1:.4f}")
    logger.info(f"New AUC: {new_auc:.4f} | Current AUC: {current_auc:.4f}")
    logger.info(f"Should promote: {should_promote}")

    ti.xcom_push(key='should_promote', value=should_promote)
    ti.xcom_push(key='comparison', value=json.dumps(comparison))


def branch_on_evaluation(**kwargs):
    """Branch: if new model passes → promote, else → rollback."""
    should_promote = kwargs['ti'].xcom_pull(key='should_promote', task_ids='evaluate_new_model')
    if should_promote:
        return 'promote_model'
    return 'rollback_model'


def promote_model(**kwargs):
    """
    Promote new model to production:
    - Update production metadata
    - Reload model in FastAPI via restart signal
    """
    import json
    import shutil
    logger = kwargs['ti'].log
    ti = kwargs['ti']

    comparison_json = ti.xcom_pull(key='comparison', task_ids='evaluate_new_model')
    comparison = json.loads(comparison_json)

    metadata = {
        'model_path': 'data/production/model.pkl',
        'vectorizer_path': 'data/production/tfidf_vectorizer.pkl',
        'promoted_at': datetime.utcnow().isoformat(),
        'metrics': {
            'val_f1_fraud': comparison['new_f1'],
            'val_roc_auc': comparison['new_auc'],
        },
        'previous_metrics': {
            'val_f1_fraud': comparison['current_f1'],
            'val_roc_auc': comparison['current_auc'],
        },
        'trigger': kwargs.get('dag_run').conf.get('trigger_reason', 'manual'),
    }

    with open(PRODUCTION_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f" Model promoted to production.")
    logger.info(f"  New F1  : {comparison['new_f1']:.4f}")
    logger.info(f"  New AUC : {comparison['new_auc']:.4f}")


def rollback_model(**kwargs):
    """
    Rollback: restore backed up production model if new model is worse.
    Also restore original train.csv.
    """
    import shutil
    import os
    logger = kwargs['ti'].log

    backup_dir = '/opt/airflow/data/production/backup'
    files_to_restore = ['model.pkl', 'tfidf_vectorizer.pkl', 'metadata.json']

    for fname in files_to_restore:
        src = os.path.join(backup_dir, fname)
        dst = os.path.join('/opt/airflow/data/production', fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            logger.info(f"Restored: {src} → {dst}")

    # Restore original train.csv
    if os.path.exists(TRAIN_DATA_PATH + '.bak'):
        shutil.copy(TRAIN_DATA_PATH + '.bak', TRAIN_DATA_PATH)
        logger.info("Restored original train.csv from backup.")

    logger.warning("New model did NOT meet quality gates. Production model rolled back.")


def verify_api_health(**kwargs):
    """
    Verify the FastAPI service is still healthy after model update.
    Fails if API is not responding or model is not loaded.
    """
    import requests
    logger = kwargs['ti'].log

    logger.info(f"Checking API health at {API_HEALTH_URL}...")

    try:
        response = requests.get(API_HEALTH_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f" API is healthy: {data}")
        else:
            raise ValueError(f"API returned status {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Cannot connect to API at {API_HEALTH_URL}. Is the container running?")


def log_retraining_summary(**kwargs):
    """Log a full retraining summary for audit purposes."""
    import json
    logger = kwargs['ti'].log
    ti = kwargs['ti']

    comparison_json = ti.xcom_pull(key='comparison', task_ids='evaluate_new_model')
    comparison = json.loads(comparison_json) if comparison_json else {}
    labeled_count = ti.xcom_pull(key='labeled_count', task_ids='export_inference_logs') or 0
    merged_count = ti.xcom_pull(key='merged_count', task_ids='merge_datasets') or 0

    logger.info("=" * 70)
    logger.info("RETRAINING PIPELINE SUMMARY")
    logger.info(f"  Triggered at     : {datetime.utcnow().isoformat()}")
    logger.info(f"  Labeled logs used: {labeled_count}")
    logger.info(f"  Total training   : {merged_count} records")
    logger.info(f"  New F1 (fraud)   : {comparison.get('new_f1', 'N/A')}")
    logger.info(f"  New AUC          : {comparison.get('new_auc', 'N/A')}")
    logger.info(f"  Prev F1 (fraud)  : {comparison.get('current_f1', 'N/A')}")
    logger.info(f"  Prev AUC         : {comparison.get('current_auc', 'N/A')}")
    logger.info(f"  Promoted         : {comparison.get('should_promote', False)}")
    logger.info("=" * 70)


# DAG Definition 
with DAG(
    dag_id='fake_job_retraining_pipeline',
    default_args=default_args,
    description='Manual retraining pipeline — triggered by admin via UI or REST API',
    schedule_interval=None,   # Manual trigger only
    start_date=datetime(2026, 4, 19),
    catchup=False,
    tags=['retraining', 'fake-job-detector'],
    params={
        'trigger_reason': 'manual',   # Can pass reason via Airflow UI
    }
) as dag:

    t1 = PythonOperator(
        task_id='export_inference_logs',
        python_callable=export_labeled_inference_logs,
    )

    t2 = PythonOperator(
        task_id='merge_datasets',
        python_callable=merge_datasets,
    )

    t3 = PythonOperator(
        task_id='validate_retrain_data',
        python_callable=validate_retrain_data,
    )

    t4 = PythonOperator(
        task_id='backup_current_model',
        python_callable=backup_current_model,
    )

    t5 = PythonOperator(
        task_id='swap_train_data',
        python_callable=swap_train_data,
    )

    t6 = BashOperator(
        task_id='run_dvc_pipeline',
        bash_command=f'cd {PROJECT_DIR} && dvc repro --force 2>&1',
        env={
            'HOME': '/home/airflow',
            'PATH': '/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin',
        },
    )

    t7 = PythonOperator(
        task_id='evaluate_new_model',
        python_callable=evaluate_new_model,
    )

    t8 = BranchPythonOperator(
        task_id='branch_on_evaluation',
        python_callable=branch_on_evaluation,
    )

    t9_promote = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model,
    )

    t9_rollback = PythonOperator(
        task_id='rollback_model',
        python_callable=rollback_model,
    )

    t10_push = BashOperator(
        task_id='dvc_push',
        bash_command=f'cd {PROJECT_DIR} && dvc push -r airflow_remote 2>&1',
        trigger_rule='all_success',
        env={
            'HOME': '/home/airflow',
            'PATH': '/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin',
        },
    )
    t11 = PythonOperator(
        task_id='verify_api_health',
        python_callable=verify_api_health,
    )

    t12 = PythonOperator(
        task_id='log_retraining_summary',
        python_callable=log_retraining_summary,
        trigger_rule='all_done',
    )

    # Pipeline 
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8
    t8 >> [t9_promote, t9_rollback]
    t9_promote >> t10_push >> t11 >> t12
    t9_rollback >> t12