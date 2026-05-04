"""
Data Loader Module — Agriculture Crop Yield Platform
"""
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_yield.csv')


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and perform basic validation of the dataset."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        _validate(df)
        return df
    except FileNotFoundError:
        logger.error(f"Dataset not found at {path}")
        raise


def _validate(df: pd.DataFrame):
    required = ['Year', 'Crop', 'Region', 'Area', 'Annual_Rainfall',
                'Average_Temperature', 'Fertilizer_Used', 'Pesticide_Used',
                'Production', 'Yield_Per_Ha']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Dataset validation passed.")


def get_summary(df: pd.DataFrame) -> dict:
    """Return high-level summary statistics."""
    return {
        'total_records': len(df),
        'crops': df['Crop'].nunique(),
        'regions': df['Region'].nunique(),
        'year_range': (int(df['Year'].min()), int(df['Year'].max())),
        'avg_yield': round(df['Yield_Per_Ha'].mean(), 2),
        'total_production': round(df['Production'].sum(), 2),
        'best_crop': df.groupby('Crop')['Yield_Per_Ha'].mean().idxmax(),
        'best_region': df.groupby('Region')['Yield_Per_Ha'].mean().idxmax(),
    }
