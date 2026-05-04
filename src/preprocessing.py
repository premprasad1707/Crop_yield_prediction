"""
Preprocessing Module — Agriculture Crop Yield Platform
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
import joblib
import os

logger = logging.getLogger(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = df.copy()
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    df.drop_duplicates(inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Numeric imputation
    imp = SimpleImputer(strategy='median')
    df[num_cols] = imp.fit_transform(df[num_cols])

    # Categorical imputation
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = _remove_outliers(df, num_cols)
    logger.info(f"Cleaned data: {df.shape}")
    return df


def _remove_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """IQR-based outlier removal for target and key numerics."""
    key = ['Yield_Per_Ha', 'Production', 'Annual_Rainfall', 'Fertilizer_Used']
    for col in [c for c in key if c in cols]:
        Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df.reset_index(drop=True)


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns; return df + encoder map."""
    df = df.copy()
    encoders = {}
    for col in ['Crop', 'Region', 'Season']:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def scale(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScale features."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    return X_scaled, scaler


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target y."""
    FEATURES = [
        'Area', 'Annual_Rainfall', 'Average_Temperature',
        'Fertilizer_Used', 'Pesticide_Used', 'Humidity',
        'Crop_enc', 'Region_enc',
        'Rainfall_Efficiency', 'Fertilizer_Efficiency',
        'CPI', 'Yield_Per_Area'
    ]
    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats].copy()
    y = df['Yield_Per_Ha'].copy()
    return X, y
