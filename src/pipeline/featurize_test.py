import pandas as pd
import pickle
import scipy.sparse as sp
import logging
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def featurize_test():
    df = pd.read_csv('data/raw/test.csv')

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


    # Load the SAME vectorizer fitted on training data
    with open('data/processed/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    X_text = tfidf.transform(df['combined_text'])  # transform only, should not fit

    meta_cols = ['has_company_logo', 'has_questions', 'has_salary',
                 'has_company_profile', 'has_requirements', 'telecommuting', 'has_benefits']
    df[meta_cols] = df[meta_cols].fillna(0).astype(int)
    X_meta = sp.csr_matrix(df[meta_cols].values)

    X = sp.hstack([X_text, X_meta])
    y = df['fraudulent'].values

    os.makedirs('data/processed', exist_ok=True)
    sp.save_npz('data/processed/X_test.npz', X)
    pd.Series(y).to_csv('data/processed/y_test.csv', index=False)

    logger.info(f"Test feature matrix shape: {X.shape}")

if __name__ == '__main__':
    featurize_test()