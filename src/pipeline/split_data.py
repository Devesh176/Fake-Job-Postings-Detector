import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data():
    df = pd.read_csv('data/raw/fake_job_postings.csv')
    
    train, test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['fraudulent']
    )
    
    os.makedirs('data/raw', exist_ok=True)
    train.to_csv('data/raw/train.csv', index=False)
    test.to_csv('data/raw/test.csv', index=False)
    logger.info(f"Train size: {len(train)} | Test size: {len(test)}")
    logger.info(f"Train fraud rate: {train['fraudulent'].mean():.3f}")
    logger.info(f"Test fraud rate: {test['fraudulent'].mean():.3f}")

if __name__ == '__main__':
    split_data()