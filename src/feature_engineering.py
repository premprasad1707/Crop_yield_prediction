"""
Feature Engineering Module — Agriculture Crop Yield Platform
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic ratios
    df['Yield_Per_Area'] = df['Yield_Per_Ha'] / (df['Area'] + 1e-9)
    df['Rainfall_Efficiency'] = df['Annual_Rainfall'] / (df['Area'] + 1e-9)
    df['Fertilizer_Efficiency'] = df['Yield_Per_Ha'] / (df['Fertilizer_Used'] + 1e-9)

    # Excess fertilizer flag (above 75th percentile)
    q75 = df['Fertilizer_Used'].quantile(0.75)
    df['Excess_Fertilizer'] = np.where(df['Fertilizer_Used'] > q75,
                                        df['Fertilizer_Used'] - q75, 0)

    # Crop Productivity Index
    df['CPI'] = (
        (df['Yield_Per_Ha'] / (df['Area'] + 1e-9))
        + (df['Annual_Rainfall'] * 0.3)
        - (df['Excess_Fertilizer'] * 0.2)
    )

    # Seasonal indicators
    season_map = {'Kharif': 1, 'Rabi': 2, 'Zaid': 3, 'Whole Year': 4}
    if 'Season' in df.columns:
        df['Season_Num'] = df['Season'].map(season_map).fillna(0).astype(int)

    # Temperature bucket
    df['Temp_Category'] = pd.cut(df['Average_Temperature'],
                                  bins=[0, 20, 28, 35, 50],
                                  labels=['Cool', 'Optimal', 'Warm', 'Hot'])

    # Year-over-year features (per crop)
    df = df.sort_values(['Crop', 'Year'])
    df['Yield_Lag1'] = df.groupby('Crop')['Yield_Per_Ha'].shift(1)
    df['Yield_Lag1'] = df['Yield_Lag1'].fillna(df['Yield_Per_Ha'].mean())

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df
