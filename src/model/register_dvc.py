import json
import pickle
import shutil
import os

def register_model():
    # Load metrics just for reporting
    with open('data/processed/metrics.json') as f:
        metrics = json.load(f)

    # Copy model to a versioned production path
    os.makedirs('data/production', exist_ok=True)
    shutil.copy('data/processed/model.pkl', 'data/production/model.pkl')
    shutil.copy('data/processed/tfidf_vectorizer.pkl', 'data/production/tfidf_vectorizer.pkl')

    # Save production metadata
    metadata = {
        'model_path': 'data/production/model.pkl',
        'vectorizer_path': 'data/production/tfidf_vectorizer.pkl',
        'metrics': metrics,
    }
    with open('data/production/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model registered to data/production/")
    print(f"Val F1  : {metrics['val_f1_fraud']:.4f}")
    print(f"Val AUC : {metrics['val_roc_auc']:.4f}")

if __name__ == '__main__':
    register_model()