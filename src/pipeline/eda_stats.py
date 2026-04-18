import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_baselines():
    df = pd.read_csv('data/processed/train_processed.csv')

    baselines = {}

    # Numerical features
    numerical_cols = [
        'has_company_logo', 'has_questions', 'has_salary',
        'has_company_profile', 'has_requirements', 'has_benefits'
    ]
    for col in numerical_cols:
        if col in df.columns:
            baselines[col] = {
                'mean': float(df[col].mean()),
                'std':  float(df[col].std()),
                'min':  float(df[col].min()),
                'max':  float(df[col].max()),
                'q25':  float(df[col].quantile(0.25)),
                'q75':  float(df[col].quantile(0.75)),
            }

    # Categorical feature distributions
    cat_cols = ['employment_type', 'required_experience', 'required_education', 'function']
    for col in cat_cols:
        if col in df.columns:
            baselines[col] = df[col].fillna('unknown').value_counts(normalize=True).to_dict()

    # Text feature: vocabulary distribution
    vectorizer = CountVectorizer(max_features=500, stop_words='english')
    vectorizer.fit(df['combined_text'].fillna(''))
    baselines['vocabulary'] = {
        'top_500_words': vectorizer.get_feature_names_out().tolist(),
        'vocab_size': len(vectorizer.vocabulary_),
    }

    # Dataset stats
    baselines['dataset'] = {
        'total_records': int(len(df)),
        'fraud_count': int(df['fraudulent'].sum()),
        'fraud_rate': float(df['fraudulent'].mean()),
    }


    # Save 
    os.makedirs('data/baselines', exist_ok=True)
    with open('data/baselines/training_baseline.json', 'w') as f:
        json.dump(baselines, f, indent=2)

    logger.info("Baselines computed and saved to data/baselines/training_baseline.json")
    return True

if __name__ == '__main__':
    compute_baselines()