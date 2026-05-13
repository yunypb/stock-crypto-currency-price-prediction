import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings


def plot_forecast_vs_actual(results, asset_name, model_name):
    """Графики факт vs прогноз для всех горизонтов."""
    fig = make_subplots(rows=len(results), cols=1,
                        shared_xaxes=True,
                        subplot_titles=[f"Горизонт {h}" for h in results.keys()])
    for i, (h, df) in enumerate(results.items()):
        fig.add_trace(go.Scatter(x=df.index, y=df['actual'],
                                 name=f'Факт (H={h})', line=dict(color='black')),
                      row=i+1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['predicted'],
                                 name=f'Прогноз (H={h})', line=dict(color='red', dash='dot')),
                      row=i+1, col=1)
    fig.update_layout(height=900, title=f"{asset_name} - {model_name}")
    fig.show()

def plot_metrics(metrics_dict, asset_name):
    """Столбчатая диаграмма метрик по моделям (для одного актива)."""
    models = list(metrics_dict.keys())
    rmse = [metrics_dict[m]['RMSE'].get(1, None) for m in models]
    mae = [metrics_dict[m]['MAE'].get(1, None) for m in models]
    mape = [metrics_dict[m]['MAPE'].get(1, None) for m in models]

    fig = go.Figure(data=[
        go.Bar(name='RMSE', x=models, y=rmse),
        go.Bar(name='MAE', x=models, y=mae),
        go.Bar(name='MAPE, %', x=models, y=mape)
    ])
    fig.update_layout(title=f"Метрики моделей для {asset_name} (горизонт 1)")
    fig.show()

def plot_equity_curves(equity_dict, asset_name):
    """Кривые доходности стратегий."""
    fig = go.Figure()
    for model_name, bt in equity_dict.items():
        fig.add_trace(go.Scatter(x=bt.index, y=bt['equity'],
                                 name=model_name))
    fig.add_trace(go.Scatter(x=bt.index, y=bt['close'] / bt['close'].iloc[0],
                             name='Buy & Hold', line=dict(dash='dot')))
    fig.update_layout(title=f"Кривые доходности для {asset_name}")
    fig.show()