import mlflow
import mlflow.sklearn
import scipy.sparse as sp
import pandas as pd
import subprocess
import time
import os
import argparse
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Optional: let CLI control experiment when using `mlflow run`
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
if experiment_name:
    mlflow.set_experiment(experiment_name)


def train(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8):
    X = sp.load_npz("data/processed/X_train.npz")
    y = pd.read_csv("data/processed/y_train.csv").values.ravel()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Works with both mlflow run and direct execution
    run = mlflow.active_run()
    if run is None:
        run = mlflow.start_run()

    # Tags
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    mlflow.set_tag("git_commit", git_hash)
    mlflow.set_tag("dataset_version", "emscad_v1")
    mlflow.set_tag("dvc_stage", "train")

    # Parameters
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "random_state": 42,
    }
    mlflow.log_params(params)
    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_param("tfidf_ngram_range", "(1,2)")
    mlflow.log_param("class_weighting", "sample_weight_balanced")
    mlflow.log_param("validation_split", 0.2)

    # Training
    sample_weights = compute_sample_weight("balanced", y_train)
    model = GradientBoostingClassifier(**params)

    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    train_duration = time.time() - start

    # Evaluation
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Inference latency
    latency_times = []
    for _ in range(100):
        t0 = time.time()
        model.predict(X_val[0:1])  # safer slicing
        latency_times.append((time.time() - t0) * 1000)

    metrics = {
        "val_f1_fraud": f1_score(y_val, y_pred, pos_label=1),
        "val_f1_weighted": f1_score(y_val, y_pred, average="weighted"),
        "val_roc_auc": roc_auc_score(y_val, y_prob),
        "val_precision_fraud": precision_score(y_val, y_pred, pos_label=1),
        "val_recall_fraud": recall_score(y_val, y_pred, pos_label=1),
        "train_duration_seconds": train_duration,
        "avg_inference_latency_ms": float(np.mean(latency_times)),
    }
    mlflow.log_metrics(metrics)

    # Artifacts
    artifact_dir = "mlflow_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
    ax.set_title("Validation Confusion Matrix")
    confusion_path = f"{artifact_dir}/confusion_matrix.png"
    fig.savefig(confusion_path)
    mlflow.log_artifact(confusion_path)
    plt.close()

    # Class distribution
    mlflow.log_dict(
        {
            "fraud_count": int(y.sum()),
            "legitimate_count": int((y == 0).sum()),
        },
        "class_distribution.json",
    )

    # Model logging
    mlflow.sklearn.log_model(model, "model")

    print(f"\nRun ID : {run.info.run_id}")
    print(f"Val F1 : {metrics['val_f1_fraud']:.4f}")
    print(f"Val AUC : {metrics['val_roc_auc']:.4f}")
    print(f"Latency : {metrics['avg_inference_latency_ms']:.2f}ms")

    return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)

    args = parser.parse_args()
    train(
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
        args.subsample,
    )