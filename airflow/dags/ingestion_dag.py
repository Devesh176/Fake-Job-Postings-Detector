from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow/src')

from pipeline.ingest import ingest_data
from pipeline.validate import validate_schema
from pipeline.eda_stats import compute_baselines
from pipeline.clean import clean_data
from pipeline.features import engineer_features
from pipeline.export import export_features

default_args = {
    'owner': 'da5402_project',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

with DAG(
    dag_id='fake_job_ingestion_pipeline',
    default_args=default_args,
    description='End-to-end data ingestion and feature engineering pipeline',
    schedule_interval='@daily',
    start_date=datetime(2026, 4, 18),
    catchup=False,
    tags=['data-engineering', 'fake-job-detector'],
) as dag:

    t1 = PythonOperator(task_id='ingest_raw_data', python_callable=ingest_data)
    t2 = PythonOperator(task_id='validate_schema', python_callable=validate_schema)
    t3 = PythonOperator(task_id='compute_baselines', python_callable=compute_baselines)
    t4 = PythonOperator(task_id='clean_data', python_callable=clean_data)
    t5 = PythonOperator(task_id='engineer_features', python_callable=engineer_features)
    t6 = PythonOperator(task_id='export_features', python_callable=export_features)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6