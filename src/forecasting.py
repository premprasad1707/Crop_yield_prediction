"""
Forecasting Module — Agriculture Crop Yield Platform
Uses Facebook Prophet (if available) or a fallback trend model.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def forecast_yield(df: pd.DataFrame, crop: str, periods: int = 5) -> pd.DataFrame:
    """
    Forecast future yield for a given crop.
    Returns a DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    sub = df[df['Crop'] == crop].groupby('Year')['Yield_Per_Ha'].mean().reset_index()
    sub.columns = ['ds', 'y']
    sub['ds'] = pd.to_datetime(sub['ds'], format='%Y')

    if HAS_PROPHET and len(sub) >= 3:
        return _prophet_forecast(sub, periods)
    else:
        return _trend_forecast(sub, periods)


def _prophet_forecast(ts: pd.DataFrame, periods: int) -> pd.DataFrame:
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.1)
    m.fit(ts)
    future = m.make_future_dataframe(periods=periods, freq='YE')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def _trend_forecast(ts: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Simple linear trend extrapolation with ±10% CI bands."""
    years = ts['ds'].dt.year.values.astype(float)
    y     = ts['y'].values
    coeffs = np.polyfit(years, y, 1)
    poly   = np.poly1d(coeffs)

    last_year = int(years.max())
    future_years = list(years) + [last_year + i for i in range(1, periods + 1)]
    yhat = poly(future_years)
    sigma = y.std() * 1.5

    result = pd.DataFrame({
        'ds': pd.to_datetime([str(int(y)) for y in future_years]),
        'yhat': np.maximum(yhat, 0),
        'yhat_lower': np.maximum(yhat - sigma, 0),
        'yhat_upper': yhat + sigma,
    })
    return result


def all_crops_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean yield per year across all crops for a global trend chart."""
    return df.groupby('Year')['Yield_Per_Ha'].mean().reset_index()
