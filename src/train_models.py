"""
Model Training Module — Agriculture Crop Yield Platform
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
import logging
import time

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logger = logging.getLogger(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models() -> dict:
    models = {
        'Linear Regression': Ridge(alpha=10),
        'Random Forest': RandomForestRegressor(n_estimators=120, max_depth=12,
                                                min_samples_split=5, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=120, max_depth=5,
                                                        learning_rate=0.08, random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08,
                                          subsample=0.85, colsample_bytree=0.85,
                                          random_state=42, verbosity=0)
    if HAS_LGB:
        models['LightGBM'] = LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.08,
                                             num_leaves=40, random_state=42, verbosity=-1)
    return models


def train_all(X_train, y_train) -> dict:
    """Train all models, return fitted model dict."""
    models = get_models()
    trained = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)
        trained[name] = model
        logger.info(f"Trained {name} in {elapsed}s")
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name.replace(" ", "_")}.pkl'))
    return trained


def evaluate_all(trained: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all trained models on test set."""
    rows = []
    for name, model in trained.items():
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        rows.append({'Model': name, 'RMSE': round(rmse, 2),
                     'MAE': round(mae, 2), 'R2': round(r2, 4)})
    df = pd.DataFrame(rows).sort_values('R2', ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(MODELS_DIR, 'metrics.csv'), index=False)
    return df


def cross_validate_all(trained: dict, X, y, cv=5) -> pd.DataFrame:
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []
    for name, model in trained.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
        rows.append({'Model': name, 'CV_Mean_R2': round(scores.mean(), 4),
                     'CV_Std_R2': round(scores.std(), 4)})
    return pd.DataFrame(rows)


def get_best_model(trained: dict, X_test, y_test):
    """Return the model with the highest R² on test set."""
    best_name, best_r2, best_model = None, -np.inf, None
    for name, model in trained.items():
        r2 = r2_score(y_test, model.predict(X_test))
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, model
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    with open(os.path.join(MODELS_DIR, 'best_model_name.json'), 'w') as f:
        json.dump({'name': best_name, 'r2': best_r2}, f)
    logger.info(f"Best model: {best_name} (R²={best_r2:.4f})")
    return best_name, best_model


def load_best_model():
    path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': feature_names,
                           'Importance': model.feature_importances_})
        return fi.sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        fi = pd.DataFrame({'Feature': feature_names,
                           'Importance': np.abs(model.coef_)})
        return fi.sort_values('Importance', ascending=False)
    return pd.DataFrame({'Feature': feature_names, 'Importance': np.ones(len(feature_names))})
