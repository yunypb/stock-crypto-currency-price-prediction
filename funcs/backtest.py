import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from funcs.load_data import get_frequency


def walk_forward_validation(series, model_func=None, test_start="2025-01-01",
                            horizon=1, logs=False, ml_config=None, **model_kwargs):
    """
    Единая Walk-Forward валидация для статистических и ML-моделей.

    Два режима работы:
    1. Старый (model_func передан):
        - model_func(train_series, steps) -> массив прогнозов
    2. ML-режим (передан ml_config):
        - ml_config должен содержать:
            'X' : pd.DataFrame признаков (индекс – даты t, как у series)
            'y' : pd.Series целевой переменной (индекс – даты t)
            'model_class' : класс модели (например, RandomForestRegressor)
            'model_params' : dict гиперпараметров
        - на каждом окне создаётся свежая модель, обучается на X, y до окна,
          предсказывает на окне.
    """
    test_start = pd.Timestamp(test_start)

    # Частота и тестовые даты (как раньше)
    freq = get_frequency(series) if ml_config is None else get_frequency(series)
    if freq == 'B':
        test_dates = pd.date_range(test_start, series.index.max(), freq='B')
        test_dates = test_dates[test_dates.isin(series.index)]
    else:
        test_dates = pd.date_range(test_start, series.index.max(), freq='D')

    preds = []
    actuals = []
    dates = []

    total_windows = max(1, len(test_dates) // horizon)
    pbar = tqdm(total=total_windows, desc=f"  H={horizon}", unit="window")
    i = 0

    # ML-режим: извлекаем данные
    if ml_config is not None:
        X_full = ml_config['X']
        y_full = ml_config['y']
        model_class = ml_config['model_class']
        model_params = ml_config['model_params']
    else:
        train = series[:test_start - pd.Timedelta(days=1)].copy()
        test_data = series[test_start:].copy()

    while i < len(test_dates):
        current_date = test_dates[i]
        # проверка наличия даты в данных
        if ml_config is not None:
            if current_date not in X_full.index:
                i += 1
                continue
        else:
            if current_date not in series.index:
                i += 1
                continue

        end_idx = min(i + horizon, len(test_dates))
        window_dates = test_dates[i:end_idx]

        try:
            if ml_config is not None:
                train_end = window_dates[0]
                X_train = X_full[X_full.index < train_end]
                y_train = y_full[y_full.index < train_end]

                if len(X_train) == 0:
                    forecast = np.full(len(window_dates), np.nan)
                else:
                    model = model_class(**model_params)
                    model.fit(X_train, y_train)
                    X_window = X_full.loc[window_dates]
                    if len(X_window) == len(window_dates):
                        forecast = model.predict(X_window)
                        forecast = np.asarray(forecast).ravel()
                    else:
                        forecast = np.full(len(window_dates), np.nan)
            else:
                forecast_raw = model_func(train, steps=len(window_dates), **model_kwargs)
                forecast = np.asarray(forecast_raw).ravel()
                if len(forecast) != len(window_dates):
                    forecast = np.full(len(window_dates), np.nan)
        except Exception as e:
            if logs:
                print(f"[WF] Ошибка на {current_date.date()}: {e}")
            forecast = np.full(len(window_dates), np.nan)

        # Сохранение результатов
        for j, d in enumerate(window_dates):
            if ml_config is not None:
                actual = y_full.loc[d] if d in y_full.index else np.nan
            else:
                actual = series.loc[d] if d in series.index else np.nan
            preds.append(forecast[j] if j < len(forecast) else np.nan)
            actuals.append(actual)
            dates.append(d)

        if ml_config is None:
            actual_window = series.loc[series.index.isin(window_dates)]
            train = pd.concat([train, actual_window]).sort_index()

        i = end_idx
        pbar.update(1)

    pbar.close()

    results = pd.DataFrame({
        'date': dates,
        'actual': actuals,
        'predicted': preds
    }).dropna(subset=['actual'])
    results.set_index('date', inplace=True)
    return results


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Backtester:
    def __init__(self,
                 predictions_df,
                 initial_capital=100_000,
                 threshold=0.005,
                 commission=0.001):
        self.df = predictions_df.copy()
        self.initial_capital = initial_capital
        self.threshold = threshold
        self.commission = commission
        self.results = None
        self.trades = None

    def run(self):
        df = self.df.copy().sort_index()

        df['prev_actual'] = df['actual'].shift(1)
        # Относительный прогнозируемый рост (на основе прогноза и цены предыдущего дня)
        df['predicted_return'] = (df['predicted'] - df['prev_actual']) / df['prev_actual']
        # Фактическая доходность от previous -> current
        df['actual_return'] = df['actual'] / df['prev_actual'] - 1

        # Удаляем первую строку (нет цены предыдущего дня)
        df = df.dropna().copy()

        # -------------------------------------------------
        # Формирование торговых сигналов
        # -------------------------------------------------
        #  1 = сигнал на покупку (лонг), -1 = сигнал на шорт, 0 = без позиции
        df['position'] = 0
        df.loc[df['predicted_return'] > self.threshold, 'position'] = 1
        df.loc[df['predicted_return'] < -self.threshold, 'position'] = -1

        # -------------------------------------------------
        # Расчёт доходности стратегии без учёта комиссии
        # -------------------------------------------------
        df['strategy_return'] = 0.0
        # Для длинной позиции — получаем фактический рост
        long_mask = df['position'] == 1
        df.loc[long_mask, 'strategy_return'] = df.loc[long_mask, 'actual_return']
        # Для короткой позиции — доходность со знаком минус
        short_mask = df['position'] == -1
        df.loc[short_mask, 'strategy_return'] = -df.loc[short_mask, 'actual_return']

        # -------------------------------------------------
        # Учёт комиссий: только при реальных сделках (вход+выход)
        # -------------------------------------------------
        # Вычитаем 2 * commission только в тех строках, где была торговля
        df.loc[long_mask | short_mask, 'strategy_return'] -= 2 * self.commission

        # -------------------------------------------------
        # Кривая капитала стратегии
        # -------------------------------------------------
        # Мультипликативный рост (реинвестиция всего капитала)
        df['equity'] = self.initial_capital * (1 + df['strategy_return']).cumprod()

        # -------------------------------------------------
        # Buy & Hold для сравнения
        # -------------------------------------------------
        df['buy_hold_equity'] = self.initial_capital * (1 + df['actual_return']).cumprod()

        # -------------------------------------------------
        # Сохранение сделок
        # -------------------------------------------------
        trades = []
        for idx, row in df.iterrows():
            if row['position'] == 0:
                continue
            trades.append({
                'date': idx,
                'type': 'LONG' if row['position'] == 1 else 'SHORT',
                'entry_price': row['prev_actual'],
                'exit_price': row['actual'],
                'predicted_price': row['predicted'],
                'predicted_return': row['predicted_return'],
                'actual_return': row['actual_return'],
                'strategy_return': row['strategy_return'],
                'equity': row['equity']
            })
        trades_df = pd.DataFrame(trades)

        self.results = df
        self.trades = trades_df
        fig = self._plot()
        return df, trades_df, fig

    def _plot(self):
        df = self.results
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Цена и прогноз", "Капитал", "Кумулятивная доходность"),
            row_heights=[0.5, 0.25, 0.25]
        )

        # Панель 1: цена и прогноз
        fig.add_trace(go.Scatter(x=df.index, y=df['actual'], name='Факт'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['predicted'], name='Прогноз',
                                 line=dict(dash='dot')), row=1, col=1)
        # Маркеры входа лонг/шорт на фактической цене выхода
        fig.add_trace(go.Scatter(
            x=df[df['position'] == 1].index,
            y=df[df['position'] == 1]['actual'],
            mode='markers', name='LONG',
            marker=dict(symbol='triangle-up', size=8, color='green')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df[df['position'] == -1].index,
            y=df[df['position'] == -1]['actual'],
            mode='markers', name='SHORT',
            marker=dict(symbol='triangle-down', size=8, color='red')
        ), row=1, col=1)

        # Панель 2: капитал (стратегия vs buy&hold)
        fig.add_trace(go.Scatter(x=df.index, y=df['equity'], name='Стратегия'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['buy_hold_equity'], 
                                 name='Buy&Hold', line=dict(dash='dash')), row=2, col=1)

        # Панель 3: кумулятивная доходность стратегии
        cum_ret = df['equity'] / self.initial_capital - 1
        fig.add_trace(go.Scatter(x=df.index, y=cum_ret, name='Cumulative Return'), row=3, col=1)

        fig.update_layout(height=1000, title='Бэктест стратегии', template='plotly_white')
        return fig


class BacktesterReturns:
    """
    Бэктестер стратегии на основе прогнозов доходностей.
    
    predictions_df: pd.DataFrame с колонками 'actual' (фактическая доходность) и 'predicted' (прогнозная доходность).
                    Фактическая доходность за день t — это (close[t] / close[t-1] - 1).
    prices: pd.Series с ценами закрытия актива (индекс совпадает с predictions_df).
            Нужна для построения кривой Buy&Hold и проверки соответствия.
    initial_capital: начальный капитал.
    threshold: порог для сигнала (если predicted_return > threshold -> лонг, < -threshold -> шорт).
    commission: комиссия за сделку в долях от капитала (вход + выход = 2 * commission).
    """
    def __init__(self, predictions_df, prices, initial_capital=100_000, threshold=0.005, commission=0.001):
        self.df = predictions_df.copy()
        self.prices = prices.copy()
        self.initial_capital = initial_capital
        self.threshold = threshold
        self.commission = commission
        self.results = None
        self.trades = None

    def run(self):
        df = self.df.sort_index()
        prices = self.prices.reindex(df.index)   # убедимся, что цены по тем же датам

        # Проверим, что фактические доходности в df соответствуют ценам (можно пересчитать для надёжности)
        df['actual'] = prices.pct_change()   # пересчитаем, чтобы избежать нестыковок
        # Удаляем первую строку (NaN после pct_change)
        df = df.dropna()

        # Сигнал на основе прогнозной доходности
        df['position'] = 0
        df.loc[df['predicted'] > self.threshold, 'position'] = 1
        df.loc[df['predicted'] < -self.threshold, 'position'] = -1

        # Доходность стратегии: position * actual_return
        df['strategy_return'] = df['position'] * df['actual']

        # Комиссия: при каждой сделке (сигнал меняется не в 0) удерживаем комиссию за вход и выход
        # Упростим: комиссия за день = |position - previous_position| * commission (две стороны сделки)
        # Для простоты можно вычитать 2*commission, когда позиция != 0 (как в исходном Backtester)
        # Но точнее: если открываем/закрываем позицию, платим комиссию.
        # Используем упрощённый подход: если позиция не 0, вычитаем 2*commission (вход и будущий выход).
        # Это переоценит комиссию при удержании, но для сравнения стратегий допустимо.
        # Сделаем как в исходном: при trades_mask = (position != 0) вычитаем 2*commission.
        trade_mask = df['position'] != 0
        df.loc[trade_mask, 'strategy_return'] -= 2 * self.commission

        # Кривая капитала
        df['equity'] = self.initial_capital * (1 + df['strategy_return']).cumprod()

        # Buy & Hold капитала
        df['buy_hold_equity'] = self.initial_capital * (prices / prices.iloc[0])

        # Формирование журнала сделок (опционально)
        trades = []
        for idx, row in df.iterrows():
            if row['position'] == 0:
                continue
            trades.append({
                'date': idx,
                'type': 'LONG' if row['position'] == 1 else 'SHORT',
                'predicted_return': row['predicted'],
                'actual_return': row['actual'],
                'strategy_return': row['strategy_return'],
                'equity': row['equity']
            })
        trades_df = pd.DataFrame(trades)

        self.results = df
        self.trades = trades_df
        fig = self._plot()
        return df, trades_df, fig

    def _plot(self):
        df = self.results
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Прогноз vs Факт доходности", "Капитал", "Кумулятивная доходность"),
            row_heights=[0.6, 0.2, 0.2]
        )

        # Панель 1
        fig.add_trace(go.Scatter(x=df.index, y=df['actual'], name='Факт доходность'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['predicted'], name='Прогноз доходность',
                                line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df[df['position'] == 1].index,
            y=df[df['position'] == 1]['predicted'],
            mode='markers', name='LONG сигнал',
            marker=dict(symbol='triangle-up', size=14, color='green')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df[df['position'] == -1].index,
            y=df[df['position'] == -1]['predicted'],
            mode='markers', name='SHORT сигнал',
            marker=dict(symbol='triangle-down', size=14, color='red')
        ), row=1, col=1)

        # Панель 2
        fig.add_trace(go.Scatter(x=df.index, y=df['equity'], name='Стратегия'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['buy_hold_equity'],
                                name='Buy&Hold', line=dict(dash='dash')), row=2, col=1)

        # Панель 3
        cum_ret = df['equity'] / self.initial_capital - 1
        fig.add_trace(go.Scatter(x=df.index, y=cum_ret, name='Cumulative Return'), row=3, col=1)

        # -------------------- КРУПНЫЕ ШРИФТЫ --------------------
        fig.update_layout(
            height=1800,          # больше высота – больше места для подписей
            width=1400,
            font=dict(size=20),   # общий размер текста (легенда, подписи осей, ticks)
            title=dict(font=dict(size=28)),
            legend=dict(font=dict(size=18)),
            template='plotly_white',
            margin=dict(l=100, r=50, t=100, b=80)
        )

        fig.update_xaxes(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        )
        fig.update_yaxes(
            tickfont=dict(size=18),
            title_font=dict(size=22)
        )

        # Увеличиваем подписи субплотов
        for annotation in fig.layout.annotations:
            annotation.font.size = 20

        return fig