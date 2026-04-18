import mlflow
import mlflow.sklearn
import scipy.sparse as sp
import pandas as pd
import pickle
import json
import subprocess
import time
import os
import argparse
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('fake-job-detector')

def train(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8):
    X = sp.load_npz('data/processed/X_train.npz')
    y = pd.read_csv('data/processed/y_train.csv').values.ravel()

    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:

        # Tags
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode().strip()
        mlflow.set_tag('git_commit', git_hash)
        mlflow.set_tag('dataset_version', 'emscad_v1')
        mlflow.set_tag('dvc_stage', 'train')

        # Parameters
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'random_state': 42,
        }
        mlflow.log_params(params)
        mlflow.log_param('tfidf_max_features', 5000)
        mlflow.log_param('tfidf_ngram_range', '(1,2)')
        mlflow.log_param('class_weighting', 'sample_weight_balanced')
        mlflow.log_param('validation_split', 0.2)

        # Training
        sample_weights = compute_sample_weight('balanced', y_train)
        model = GradientBoostingClassifier(**params)

        start = time.time()
        model.fit(X_train, y_train, sample_weight=sample_weights)
        train_duration = time.time() - start

        # Evaluation on validation set
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Inference latency
        latency_times = []
        for _ in range(100):
            t0 = time.time()
            model.predict(X_val[0])
            latency_times.append((time.time() - t0) * 1000)

        metrics = {
            'val_f1_fraud': f1_score(y_val, y_pred, pos_label=1),
            'val_f1_weighted': f1_score(y_val, y_pred, average='weighted'),
            'val_roc_auc': roc_auc_score(y_val, y_prob),
            'val_precision_fraud': precision_score(y_val, y_pred, pos_label=1),
            'val_recall_fraud': recall_score(y_val, y_pred, pos_label=1),
            'train_duration_seconds': train_duration,
            'avg_inference_latency_ms': float(np.mean(latency_times)),
        }
        mlflow.log_metrics(metrics)

        # Artifacts
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
        ax.set_title('Validation Confusion Matrix')
        fig.savefig('/tmp/confusion_matrix.png')
        mlflow.log_artifact('/tmp/confusion_matrix.png')
        plt.close()

        mlflow.log_dict(
            {'fraud_count': int(y.sum()), 'legitimate_count': int((y == 0).sum())},
            'class_distribution.json'
        )

        mlflow.sklearn.log_model(model, 'model')

        # Save locally for DVC
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/model.pkl', 'wb') as f:
            pickle.dump(model, f)

        with open('data/processed/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nRun ID : {run.info.run_id}")
        print(f"Val F1 : {metrics['val_f1_fraud']:.4f}")
        print(f"Val AUC : {metrics['val_roc_auc']:.4f}")
        print(f"Latency : {metrics['avg_inference_latency_ms']:.2f}ms")

        return run.info.run_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=0.8)
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth, args.learning_rate, args.subsample)