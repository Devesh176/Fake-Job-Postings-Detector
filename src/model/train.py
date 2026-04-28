import os
import time
import json
import pickle
import argparse
import subprocess
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

def train(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, use_mlflow=False):
    # THE SHARED TRUNK: Data & Core Math
    X = sp.load_npz("data/processed/X_train.npz")
    y = pd.read_csv("data/processed/y_train.csv").values.ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "random_state": 42,
    }

    # Pure Python gentle penalty (NO NUMPY)
    sample_weights = [5.0 if val == 1 else 1.0 for val in y_train]
    
    model = GradientBoostingClassifier(**params)

    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    train_duration = time.time() - start

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Inference latency (Pure Python average, NO NUMPY)
    latency_times = []
    for _ in range(100):
        t0 = time.time()
        model.predict(X_val[0:1]) 
        latency_times.append((time.time() - t0) * 1000)
    
    avg_latency = sum(latency_times) / len(latency_times)

    metrics = {
        "val_f1_fraud": f1_score(y_val, y_pred, pos_label=1),
        "val_f1_weighted": f1_score(y_val, y_pred, average="weighted"),
        "val_roc_auc": roc_auc_score(y_val, y_prob),
        "val_precision_fraud": precision_score(y_val, y_pred, pos_label=1),
        "val_recall_fraud": recall_score(y_val, y_pred, pos_label=1),
        "train_duration_seconds": train_duration,
        "avg_inference_latency_ms": float(avg_latency),
    }

    # 2. THE DEPLOYMENT FORK
    if use_mlflow:
        print("MLflow flag detected. Logging and registering model...")
        
        # Check if we are being executed by `mlflow run` CLI
        in_mlflow_run = "MLFLOW_RUN_ID" in os.environ

        # Only do manual setup if we are NOT running via `mlflow run`
        if not in_mlflow_run:
            MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fake-job-detector")
            client = mlflow.tracking.MlflowClient()
            try:
                exp = client.get_experiment_by_name(experiment_name)
                if exp is None:
                    client.create_experiment(
                        name=experiment_name,
                        artifact_location=f"/home/guest/tools/Fake-Job-Postings-Detector/mlflow_artifacts/{experiment_name}"
                    )
            except Exception as e:
                pass
                
            mlflow.set_experiment(experiment_name)
        
        # Use active_run() so it smoothly attaches to the CLI run if it exists
        with mlflow.start_run() as run:
            try:
                git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
                mlflow.set_tag("git_commit", git_hash)
            except Exception as e:
                print(f"Git hash logging skipped: {e}") 
                
            mlflow.set_tag("dataset_version", "emscad_v1")
            mlflow.set_tag("dvc_stage", "train")

            mlflow.log_params(params)
            mlflow.log_param("class_weighting", "gentle_5_1_penalty")
            mlflow.log_metrics(metrics)

            artifact_dir = "mlflow_artifacts"
            os.makedirs(artifact_dir, exist_ok=True)
            
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
            ax.set_title("Validation Confusion Matrix")
            confusion_path = f"{artifact_dir}/confusion_matrix.png"
            fig.savefig(confusion_path)
            mlflow.log_artifact(confusion_path)
            plt.close()

            fraud_count = sum(1 for val in y if val == 1)
            legit_count = len(y) - fraud_count
            mlflow.log_dict({"fraud_count": fraud_count, "legitimate_count": legit_count}, "class_distribution.json")

            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="model",
                registered_model_name="FakeJobDetector_GBC"
            )
            
            print(f"Registered in MLflow! Run ID: {run.info.run_id}")    

    else:
        print("Standard DVC run. Saving artifacts to local disk...")
        os.makedirs('data/processed', exist_ok=True)
        
        with open('data/processed/model.pkl', 'wb') as f:
            pickle.dump(model, f)

        with open('data/processed/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Local artifacts saved. Val F1: {metrics['val_f1_fraud']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--use-mlflow", action="store_true", help="Log and register model in MLflow")

    args = parser.parse_args()
    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        use_mlflow=args.use_mlflow
    )