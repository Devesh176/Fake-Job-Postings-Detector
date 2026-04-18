
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import pickle
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data():
    df = pd.read_csv('data/raw/train.csv')

    # Combine text fields
    df['combined_text'] = (
        df['title'].fillna('') + ' ' +
        df['location'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['company_profile'].fillna('') + ' ' +
        df['requirements'].fillna('') + ' ' +
        df['benefits'].fillna('')
    )

    # Binary metadata flags
    df['has_salary'] = df['salary_range'].notna().astype(int)
    df['has_company_profile'] = (df['company_profile'].fillna('') != '').astype(int)
    df['has_requirements'] = (df['requirements'].fillna('') != '').astype(int)
    df['has_benefits'] = (df['benefits'] != '').astype(int)

    # TF-IDF on combined text
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
    X_text = tfidf.fit_transform(df['combined_text'])

    # Metadata features
    meta_cols = ['has_company_logo', 'has_questions', 'has_salary',
                 'has_company_profile', 'has_requirements', 'telecommuting', 'has_benefits']
    df[meta_cols] = df[meta_cols].fillna(0).astype(int)
    X_meta = sp.csr_matrix(df[meta_cols].values)

    # Combine text + metadata
    X = sp.hstack([X_text, X_meta])
    y = df['fraudulent'].values
    
    # Save artifacts
    os.makedirs('data/processed', exist_ok=True)
    sp.save_npz('data/processed/X_train.npz', X)
    pd.Series(y).to_csv('data/processed/y_train.csv', index=False)

    # Save vectorizer — MUST be reused at inference time
    with open('data/processed/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    # Save processed dataframe for eda_stats
    df.to_csv('data/processed/train_processed.csv', index=False)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Fraud rate: {y.mean():.3f}")

if __name__ == '__main__':
    preprocess_data()