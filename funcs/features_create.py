"""
Функции для машинного обучения:
- генерация признаков (лаги, SMA, EMA, волатильность, моментум, день недели)
- Walk-Forward валидация для ML-моделей
- обёртки для RandomForest и CatBoost, возвращающие прогноз
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
import warnings


def create_features(series, horizon=1):
    """
    Создаёт признаки для прогноза close[t+horizon] на основе данных до момента t.
    Возвращает:
        X : pd.DataFrame (индекс = даты t, колонки — признаки)
        y : pd.Series  (индекс = даты t, значения = close[t+horizon])
    """
    s = series.copy()
    df = pd.DataFrame(index=s.index)
    df['close'] = s

    # Лаги цены
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'close_lag{lag}'] = s.shift(lag)

    # Лаги доходностей (1 день)
    df['return_lag1'] = s.pct_change().shift(1)
    df['return_lag2'] = s.pct_change().shift(2)
    df['return_lag3'] = s.pct_change().shift(3)

    # Скользящие средние и отношения
    for w in [5, 10, 20]:
        sma = s.rolling(window=w, min_periods=1).mean()
        df[f'sma{w}'] = sma
        df[f'ratio_sma{w}'] = s / sma

        ema = s.ewm(span=w, adjust=False).mean()
        df[f'ema{w}'] = ema
        df[f'ratio_ema{w}'] = s / ema

    # Волатильность (с min_periods=1, чтобы избежать лишних NaN)
    for w in [5, 10, 20]:
        df[f'vol{w}'] = s.pct_change().rolling(window=w, min_periods=1).std()

    # Моментум (изменение за w дней)
    for w in [5, 10, 20]:
        df[f'momentum{w}'] = s.pct_change(periods=w)

    # Абсолютные изменения
    for w in [1, 5, 10]:
        df[f'diff{w}'] = s.diff(w)

    # День недели (one-hot)
    dow = s.index.dayofweek
    for d in range(7):
        df[f'dow{d}'] = (dow == d).astype(int)

    # Экстремумы скользящего окна (по close)
    for w in [5, 10, 20]:
        df[f'close_max{w}'] = s.rolling(window=w, min_periods=1).max()
        df[f'close_min{w}'] = s.rolling(window=w, min_periods=1).min()

    # Целевая переменная: close[t+horizon]
    df['target'] = s.shift(-horizon)

    # Удаляем все строки, где есть NaN (и в признаках, и в цели)
    df.dropna(inplace=True)

    # Разделяем обратно
    X = df.drop('target', axis=1)
    y = df['target']

    return X, y



def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI по Уайлдеру.
    Используем только прошлые и текущие значения цены, без leakage.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_dl_feature_frame(series: pd.Series, horizon: int = 1) -> pd.DataFrame:
    """
    Создаёт фичи для LSTM / 1D-CNN.

    Возвращает DataFrame:
        index  = даты t
        columns = признаки + target
    где target = close[t + horizon]
    """
    s = series.astype(float).copy()
    df = pd.DataFrame(index=s.index)

    df["close"] = s
    df["log_close"] = np.log(s)

    # Доходность 1 дня
    df["logret_1"] = df["log_close"].diff()

    # Лаги доходности
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"logret_lag_{lag}"] = df["logret_1"].shift(lag)

    # Лаги цены
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"close_lag_{lag}"] = s.shift(lag)

    # Скользящие окна: тренд, волатильность, уровни
    windows = [5, 10, 20, 60]

    for w in windows:
        sma = s.rolling(window=w, min_periods=w).mean()
        ema = s.ewm(span=w, adjust=False).mean()
        std = s.rolling(window=w, min_periods=w).std(ddof=0)

        rolling_max = s.rolling(window=w, min_periods=w).max()
        rolling_min = s.rolling(window=w, min_periods=w).min()

        ret_mean = df["logret_1"].rolling(window=w, min_periods=w).mean()
        ret_std = df["logret_1"].rolling(window=w, min_periods=w).std(ddof=0)

        df[f"sma_{w}"] = sma
        df[f"ema_{w}"] = ema

        df[f"price_vs_sma_{w}"] = s / sma - 1
        df[f"price_vs_ema_{w}"] = s / ema - 1

        df[f"zscore_{w}"] = (s - sma) / std.replace(0, np.nan)

        df[f"ret_mean_{w}"] = ret_mean
        df[f"vol_{w}"] = ret_std

        df[f"trend_{w}"] = s / sma - 1
        df[f"drawdown_{w}"] = s / rolling_max - 1
        df[f"range_pos_{w}"] = (s - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)

        df[f"momentum_{w}"] = s / s.shift(w) - 1

        upper = sma + 2 * std
        lower = sma - 2 * std
        df[f"bb_width_{w}"] = (upper - lower) / sma.replace(0, np.nan)
        df[f"bb_pos_{w}"] = (s - lower) / (upper - lower).replace(0, np.nan)

        df[f"skew_{w}"] = df["logret_1"].rolling(window=w, min_periods=w).skew()
        df[f"kurt_{w}"] = df["logret_1"].rolling(window=w, min_periods=w).kurt()

    # Простые режимные признаки
    df["vol_ratio_5_20"] = df["vol_5"] / df["vol_20"].replace(0, np.nan)
    df["vol_ratio_10_60"] = df["vol_10"] / df["vol_60"].replace(0, np.nan)
    df["sma_5_20_spread"] = df["sma_5"] / df["sma_20"] - 1
    df["ema_5_20_spread"] = df["ema_5"] / df["ema_20"] - 1

    # RSI
    df["rsi_7"] = _rsi(s, period=7)
    df["rsi_14"] = _rsi(s, period=14)

    # Календарные признаки
    dow = s.index.dayofweek
    month = s.index.month

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    df["is_month_start"] = s.index.is_month_start.astype(int)
    df["is_month_end"] = s.index.is_month_end.astype(int)

    # Target: будущая цена
    df["target"] = s.shift(-horizon)

    df = df.dropna().copy()
    return df


def create_features_returns(close_series, horizon=1):
    """
    Признаки на основе доходностей и календаря.
    Возвращает X и y с DatetimeIndex.
    """
    df = pd.DataFrame(index=close_series.index)
    df['close'] = close_series

    # Дневная доходность
    df['ret1'] = df['close'].pct_change()

    # Лаги доходностей
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'ret_lag{lag}'] = df['ret1'].shift(lag)

    # Волатильность и скользящие средние
    df['volatility_20'] = df['ret1'].rolling(20).std()
    df['ret_ma5'] = df['ret1'].rolling(5).mean()
    df['ret_ma20'] = df['ret1'].rolling(20).mean()

    # Календарные признаки (OHE) вручную
    for d in range(7):
        df[f'dow_{d}'] = (df.index.dayofweek == d).astype(float)
    for m in range(12):
        df[f'month_{m}'] = (df.index.month - 1 == m).astype(float)

    # Целевая переменная
    df['future_ret'] = df['close'].pct_change(horizon).shift(-horizon)

    # Удаляем строки с NaN (из-за rolling и shift)
    df.dropna(inplace=True)

    # Признаки – все, кроме close, ret1, future_ret
    feature_cols = [c for c in df.columns if c not in ['close', 'ret1', 'future_ret']]
    X = df[feature_cols]
    y = df['future_ret']

    return X, y