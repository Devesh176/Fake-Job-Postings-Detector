import scipy.sparse as sp
import pandas as pd
import pickle
import json
import os
import argparse
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split

def train(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8):
    X = sp.load_npz('data/processed/X_train.npz')
    y = pd.read_csv('data/processed/y_train.csv').values.ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        'n_estimators':  n_estimators,
        'max_depth':     max_depth,
        'learning_rate': learning_rate,
        'subsample':     subsample,
        'random_state':  42,
    }

    # sample_weights = compute_sample_weight('balanced', y_train)
    # gentle 5:1 penalty:
    sample_weights = np.where(y_train == 1, 5.0, 1.0)
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        'val_f1_fraud':        f1_score(y_val, y_pred, pos_label=1),
        'val_f1_weighted':     f1_score(y_val, y_pred, average='weighted'),
        'val_roc_auc':         roc_auc_score(y_val, y_prob),
        'val_precision_fraud': precision_score(y_val, y_pred, pos_label=1),
        'val_recall_fraud':    recall_score(y_val, y_pred, pos_label=1),
    }

    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('data/processed/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Val F1    : {metrics['val_f1_fraud']:.4f}")
    print(f"Val AUC   : {metrics['val_roc_auc']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators',  type=int,   default=300)
    parser.add_argument('--max_depth',     type=int,   default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--subsample',     type=float, default=0.8)
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth, args.learning_rate, args.subsample)