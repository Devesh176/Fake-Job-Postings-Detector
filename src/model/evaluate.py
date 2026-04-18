import scipy.sparse as sp
import pandas as pd
import pickle
import json
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, classification_report
)

def evaluate():
    X_test = sp.load_npz('data/processed/X_test.npz')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

    with open('data/processed/model.pkl', 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'f1_fraud': f1_score(y_test, y_pred, pos_label=1),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'precision_fraud': precision_score(y_test, y_pred, pos_label=1),
        'recall_fraud': recall_score(y_test, y_pred, pos_label=1),
    }

    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    with open('data/processed/eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == '__main__':
    evaluate()