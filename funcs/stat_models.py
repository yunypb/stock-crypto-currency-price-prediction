import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
# (остальные импорты, если есть, оставьте)

def sma_predict(train_series, steps, window=20):
    preds = []
    hist = train_series.copy()
    for _ in range(steps):
        if len(hist) >= window:
            next_val = hist.iloc[-window:].mean()
        else:
            next_val = hist.mean()
        preds.append(next_val)
        hist = pd.concat([hist, pd.Series([next_val], index=[hist.index[-1] + pd.Timedelta(days=1)])])
    return np.array(preds)

def ets_predict(train_series, steps, seasonal_periods):
    model = ExponentialSmoothing(
        train_series,
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods,
        damped_trend=True,
        initialization_method='estimated'
    )
    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(steps)
    return np.asarray(forecast).ravel()

def sarima_predict(train_series, steps, seasonal_periods):
    """SARIMA с автоматическим выбором порядка – ускоренный вариант."""
    model = pm.auto_arima(
        train_series,
        seasonal=True,
        m=seasonal_periods,
        max_p=2,
        max_q=2,
        max_P=2,
        max_Q=2,
        max_d=1,
        max_D=1,
        stepwise=False,
        n_jobs=-1,
        suppress_warnings=True,
        error_action='ignore',
        trace=False,
    )
    forecast = model.predict(n_periods=steps)
    return np.asarray(forecast).ravel()