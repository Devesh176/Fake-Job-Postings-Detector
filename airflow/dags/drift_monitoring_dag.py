"""
DAG 1: Daily Drift Monitoring Pipeline
Runs daily, computes PSI drift scores against training baselines,
sends email alerts via SMTP (Mailtrap) if drift is detected,
and logs a daily summary report.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
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
    'email_on_failure': False,  # We handle email manually via tasks
    'email_on_retry': False,
}

BASELINE_PATH = '/opt/airflow/data/baselines/training_baseline.json'
DRIFT_REPORT_PATH = '/opt/airflow/data/baselines/drift_report.json'
PSI_THRESHOLD = float(os.getenv('PSI_THRESHOLD', 0.2))

# Task Functions 

def fetch_recent_inference_logs(**kwargs):
    """
    Fetch inference logs from the past 24 hours from PostgreSQL.
    Pushes the data to XCom for downstream tasks.
    """
    import sqlite3
    import pandas as pd
    import json
    from datetime import datetime, timedelta

    logger = kwargs['ti'].log
    logger.info("Fetching recent inference logs from SQLite...")

    DB_PATH = '/opt/airflow/data/inference.db'
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    cur = conn.execute("""
        SELECT
            title, description, company_profile, requirements,
            employment_type, has_company_logo, has_questions,
            salary_range, prediction, fraud_probability,
            confidence, inference_latency_ms, user_feedback,
            timestamp
        FROM inference_log
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
    """, (since,))

    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    conn.close()

    df = pd.DataFrame(rows, columns=columns)
    record_count = len(df)
    logger.info(f"Fetched {record_count} inference records from last 24 hours.")

    if record_count == 0:
        logger.warning("No inference logs found in last 24 hours. Drift check skipped.")
        kwargs['ti'].xcom_push(key='record_count', value=0)
        kwargs['ti'].xcom_push(key='inference_data', value=json.dumps([]))
        return

    kwargs['ti'].xcom_push(key='record_count', value=record_count)
    kwargs['ti'].xcom_push(key='inference_data', value=df.to_json(orient='records'))
    logger.info("Inference data pushed to XCom.")


def compute_drift(**kwargs):
    """
    Compute PSI (Population Stability Index) for each monitored feature
    by comparing recent inference data against training baselines.
    Saves drift report to DRIFT_REPORT_PATH.
    """
    import json
    import numpy as np
    import pandas as pd

    logger = kwargs['ti'].log
    ti = kwargs['ti']

    record_count = ti.xcom_pull(key='record_count', task_ids='fetch_inference_logs')
    if record_count == 0:
        logger.warning("No data to compute drift. Skipping.")
        ti.xcom_push(key='drift_detected', value=False)
        ti.xcom_push(key='drift_report', value=json.dumps({}))
        return

    inference_json = ti.xcom_pull(key='inference_data', task_ids='fetch_inference_logs')
    df = pd.read_json(inference_json)

    with open(BASELINE_PATH, 'r') as f:
        baselines = json.load(f)

    def compute_psi(expected_dist: dict, actual_dist: dict, epsilon=1e-6) -> float:
        all_categories = set(expected_dist.keys()) | set(actual_dist.keys())
        psi = 0.0
        for cat in all_categories:
            e = expected_dist.get(cat, epsilon)
            a = actual_dist.get(cat, epsilon)
            e = max(e, epsilon)
            a = max(a, epsilon)
            psi += (a - e) * np.log(a / e)
        return round(float(psi), 4)

    drift_report = {
        'computed_at': datetime.utcnow().isoformat(),
        'records_analyzed': record_count,
        'psi_threshold': PSI_THRESHOLD,
        'features': {},
        'drift_detected': False,
        'drifted_features': [],
    }

    # Categorical feature drift 
    cat_cols = ['employment_type']
    for col in cat_cols:
        if col in df.columns and col in baselines:
            actual_dist = df[col].fillna('unknown').value_counts(normalize=True).to_dict()
            expected_dist = baselines[col]
            psi = compute_psi(expected_dist, actual_dist)
            drifted = psi >= PSI_THRESHOLD
            drift_report['features'][col] = {
                'psi': psi,
                'drifted': drifted,
                'actual_distribution': actual_dist,
                'expected_distribution': expected_dist,
            }
            if drifted:
                drift_report['drifted_features'].append(col)
            logger.info(f"PSI [{col}]: {psi} {'DRIFT' if drifted else 'OK'}")

    # Numerical feature drift 
    num_cols = ['has_company_logo', 'has_questions']
    for col in num_cols:
        if col in df.columns and col in baselines:
            actual_mean = float(df[col].fillna(0).mean())
            expected_mean = baselines[col]['mean']
            expected_std = baselines[col]['std'] or 1e-6
            z_score = abs(actual_mean - expected_mean) / expected_std
            drifted = z_score > 2.0  # >2 std deviations = drift
            drift_report['features'][col] = {
                'actual_mean': round(actual_mean, 4),
                'expected_mean': round(expected_mean, 4),
                'z_score': round(z_score, 4),
                'drifted': drifted,
            }
            if drifted:
                drift_report['drifted_features'].append(col)
            logger.info(f"Z-score [{col}]: {z_score:.4f} {'DRIFT' if drifted else 'OK'}")

    # Prediction distribution drift
    if 'prediction' in df.columns:
        actual_fraud_rate = float((df['prediction'] == 'Fraudulent').mean())
        expected_fraud_rate = baselines.get('dataset', {}).get('fraud_rate', 0.048)
        deviation = abs(actual_fraud_rate - expected_fraud_rate)
        drifted = deviation > 0.15  # >15% shift in fraud rate
        drift_report['features']['fraud_rate'] = {
            'actual_fraud_rate': round(actual_fraud_rate, 4),
            'expected_fraud_rate': round(expected_fraud_rate, 4),
            'deviation': round(deviation, 4),
            'drifted': drifted,
        }
        if drifted:
            drift_report['drifted_features'].append('fraud_rate')
        logger.info(f"Fraud rate drift: {deviation:.4f} {'DRIFT' if drifted else 'OK'}")

    # Latency stats
    if 'inference_latency_ms' in df.columns:
        latencies = df['inference_latency_ms'].dropna()
        drift_report['latency_stats'] = {
            'avg_ms': round(float(latencies.mean()), 2),
            'p95_ms': round(float(latencies.quantile(0.95)), 2),
            'max_ms': round(float(latencies.max()), 2),
            'sla_breach': float(latencies.quantile(0.95)) > 200,
        }

    drift_report['drift_detected'] = len(drift_report['drifted_features']) > 0

    # Save report
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    with open(DRIFT_REPORT_PATH, 'w') as f:
        json.dump(drift_report, f, indent=2)

    ti.xcom_push(key='drift_detected', value=drift_report['drift_detected'])
    ti.xcom_push(key='drift_report', value=json.dumps(drift_report))
    logger.info(f"Drift report saved. Drift detected: {drift_report['drift_detected']}")


def branch_on_drift(**kwargs):
    """Branch: if drift detected -> send alert, else -> log ok."""
    drift_detected = kwargs['ti'].xcom_pull(key='drift_detected', task_ids='compute_drift')
    if drift_detected:
        return 'send_drift_alert'
    return 'log_no_drift'


def build_drift_alert_email(**kwargs):
    """Build the HTML drift alert email body and push to XCom."""
    import json
    ti = kwargs['ti']
    report_json = ti.xcom_pull(key='drift_report', task_ids='compute_drift')
    report = json.loads(report_json)

    drifted = report.get('drifted_features', [])
    features_html = ""
    for feat, info in report.get('features', {}).items():
        is_drifted = info.get('drifted', False)
        color = "#ff4d6d" if is_drifted else "#00e5a0"
        status = "⚠️ DRIFT" if is_drifted else "✓ OK"
        detail = ""
        if 'psi' in info:
            detail = f"PSI: {info['psi']}"
        elif 'z_score' in info:
            detail = f"Z-score: {info['z_score']}"
        elif 'deviation' in info:
            detail = f"Deviation: {info['deviation']}"
        features_html += f"""
        <tr>
          <td style="padding:10px; border-bottom:1px solid #eee;">{feat}</td>
          <td style="padding:10px; border-bottom:1px solid #eee; color:{color}; font-weight:bold;">{status}</td>
          <td style="padding:10px; border-bottom:1px solid #eee; font-family:monospace;">{detail}</td>
        </tr>"""

    latency = report.get('latency_stats', {})
    latency_html = f"""
        <p>Avg: {latency.get('avg_ms', 'N/A')}ms |
           P95: {latency.get('p95_ms', 'N/A')}ms |
           Max: {latency.get('max_ms', 'N/A')}ms |
           SLA Breach: {'Yes ⚠️' if latency.get('sla_breach') else 'No ✓'}
        </p>""" if latency else ""

    email_body = f"""
    <html><body style="font-family:sans-serif; color:#333; max-width:600px; margin:0 auto;">
      <div style="background:#ff4d6d; padding:24px; border-radius:8px 8px 0 0;">
        <h1 style="color:white; margin:0;">⚠️ Data Drift Detected</h1>
        <p style="color:rgba(255,255,255,0.8); margin:8px 0 0;">
          Fake Job Detector — Daily Drift Report
        </p>
      </div>
      <div style="background:#f9f9f9; padding:24px; border-radius:0 0 8px 8px; border:1px solid #eee;">
        <p>Drift has been detected in the following features. Consider triggering the retraining DAG.</p>
        <h3>Drifted Features: {', '.join(drifted)}</h3>
        <table style="width:100%; border-collapse:collapse;">
          <thead>
            <tr style="background:#f0f0f0;">
              <th style="padding:10px; text-align:left;">Feature</th>
              <th style="padding:10px; text-align:left;">Status</th>
              <th style="padding:10px; text-align:left;">Score</th>
            </tr>
          </thead>
          <tbody>{features_html}</tbody>
        </table>
        <h3>Latency Stats</h3>
        {latency_html}
        <h3>Summary</h3>
        <p>Records analyzed: <strong>{report.get('records_analyzed', 0)}</strong></p>
        <p>Computed at: <strong>{report.get('computed_at', '')}</strong></p>
        <p>PSI Threshold: <strong>{report.get('psi_threshold', 0.2)}</strong></p>
        <hr style="border:none; border-top:1px solid #eee; margin:20px 0;">
        <p style="color:#999; font-size:12px;">
          This is an automated alert from the Fake Job Detector MLOps pipeline.
          To retrain the model, trigger the <strong>fake_job_retraining_pipeline</strong> DAG in Airflow.
        </p>
      </div>
    </html></body>
    """

    ti.xcom_push(key='alert_email_body', value=email_body)
    kwargs['ti'].log.info("Drift alert email body built.")


def log_no_drift(**kwargs):
    """Log a clean daily summary when no drift is detected."""
    import json
    ti = kwargs['ti']
    report_json = ti.xcom_pull(key='drift_report', task_ids='compute_drift')
    report = json.loads(report_json) if report_json else {}
    logger = kwargs['ti'].log
    logger.info("=" * 60)
    logger.info("✓ DAILY DRIFT CHECK PASSED — No drift detected.")
    logger.info(f"  Records analyzed : {report.get('records_analyzed', 0)}")
    logger.info(f"  Computed at      : {report.get('computed_at', 'N/A')}")
    for feat, info in report.get('features', {}).items():
        if 'psi' in info:
            logger.info(f"  [{feat}] PSI: {info['psi']}")
        elif 'z_score' in info:
            logger.info(f"  [{feat}] Z-score: {info['z_score']}")
    logger.info("=" * 60)


def send_daily_log(**kwargs):
    """Log a daily summary regardless of drift outcome."""
    import json
    ti = kwargs['ti']
    report_json = ti.xcom_pull(key='drift_report', task_ids='compute_drift')
    report = json.loads(report_json) if report_json else {}
    logger = kwargs['ti'].log

    logger.info("=" * 60)
    logger.info(f"DAILY DRIFT SUMMARY — {datetime.utcnow().strftime('%Y-%m-%d')}")
    logger.info(f"Records analyzed  : {report.get('records_analyzed', 0)}")
    logger.info(f"Drift detected    : {report.get('drift_detected', False)}")
    logger.info(f"Drifted features  : {report.get('drifted_features', [])}")
    logger.info(f"Drift report saved: {DRIFT_REPORT_PATH}")
    logger.info("=" * 60)


# ── DAG Definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id='fake_job_drift_monitoring',
    default_args=default_args,
    description='Daily drift detection pipeline with SMTP alerts via Mailtrap',
    schedule_interval='@daily',
    start_date=datetime(2026, 4, 19),
    catchup=False,
    tags=['monitoring', 'drift', 'fake-job-detector'],
) as dag:

    t1 = PythonOperator(
        task_id='fetch_inference_logs',
        python_callable=fetch_recent_inference_logs,
    )

    t2 = PythonOperator(
        task_id='compute_drift',
        python_callable=compute_drift,
    )

    t3 = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=branch_on_drift,
    )

    t4_build = PythonOperator(
        task_id='build_drift_alert_email',
        python_callable=build_drift_alert_email,
    )

    t4_alert = EmailOperator(
        task_id='send_drift_alert',
        to='{{ var.value.get("alert_email", "admin@example.com") }}',
        subject='[JobGuard] Data Drift Detected — {{ ds }}',
        html_content="{{ ti.xcom_pull(task_ids='build_drift_alert_email', key='alert_email_body') }}",
        conn_id='smtp_mailtrap',
    )

    t4_no_drift = PythonOperator(
        task_id='log_no_drift',
        python_callable=log_no_drift,
    )

    t5 = PythonOperator(
        task_id='send_daily_log',
        python_callable=send_daily_log,
        trigger_rule='none_failed_min_one_success',
    )

    # Pipeline 
    t1 >> t2 >> t3 >> [t4_build, t4_no_drift]
    t4_build >> t4_alert >> t5
    t4_no_drift >> t5