"""
Evaluate Module — Agriculture Crop Yield Platform
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))
    mape  = float(np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100)
    return {'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
            'R2': round(r2, 4), 'MAPE': round(mape, 2)}


def residuals(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    preds = model.predict(X_test)
    return pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': preds,
        'Residual': y_test.values - preds
    })
