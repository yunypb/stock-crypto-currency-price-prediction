import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings



def evaluate_metrics(df, skip_mape=False):
    """Вычисляет RMSE, MAE, MAPE (опционально) для каждого горизонта по словарю результатов."""
    metrics = {}
    y_true = df['actual'].values
    y_pred = df['predicted'].values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        metrics = {'RMSE': np.nan, 'MAE': np.nan}
        if not skip_mape:
            metrics['MAPE'] = np.nan
        return metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    metrics = {'RMSE': rmse, 'MAE': mae}
    if not skip_mape:
        # Чтобы избежать inf при нулевом actual, можно добавить проверку, но проще просто пропустить
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['MAPE'] = mape
    return metrics

