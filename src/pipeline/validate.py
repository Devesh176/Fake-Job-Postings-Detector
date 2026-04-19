import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'title', 'location', 'description', 'company_profile', 'requirements', 'benefits', 'telecommuting',
    'employment_type', 'has_company_logo', 'has_questions', 'required_experience', 'required_education',
    'industry', 'function', 'fraudulent'
]

def validate_schema():
    df = pd.read_csv('data/raw/train.csv')

    # Check 1 — Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    logger.info("Check 1 passed: All required columns present")

    # Check 2 — Target variable is binary
    assert set(df['fraudulent'].dropna().unique()).issubset({0, 1}), \
        "Target variable has unexpected values"
    logger.info("Check 2 passed: Target variable is binary")

    # Check 3 — Critical columns not too empty
    for col in ['title', 'description', 'fraudulent']:
        missing_pct = df[col].isnull().mean()
        if missing_pct > 0.3:
            raise ValueError(f"Column '{col}' has {missing_pct:.1%} missing — too high")
    logger.info("Check 3 passed: No critical columns are excessively empty")

    # Check 4 — No duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"{duplicates} duplicate rows found — dropping them")
        df = df.drop_duplicates()

    # Check 5 — Class imbalance warning
    fraud_rate = df['fraudulent'].mean()
    logger.info(f"Fraud rate: {fraud_rate:.3f} ({fraud_rate*100:.1f}%)")
    if fraud_rate < 0.02:
        logger.warning("Severe class imbalance detected — ensure class weights are used")

    logger.info(f"Validation passed. Shape: {df.shape}")
    return True

if __name__ == '__main__':
    validate_schema()