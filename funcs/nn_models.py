import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------
# 1. Вспомогательные утилиты
# ---------------------------------------------------------------------
def infer_frequency(index):
    """
    Определяет частоту индекса: 'B' для бизнес-дней (пропуски выходных),
    'D' для непрерывных дней (криптовалюты).
    """
    diffs = index.to_series().diff().dropna().dt.days
    if (diffs > 1).any() or (diffs == 3).any():
        return 'B'
    else:
        return 'D'

def make_features(series, lookback):
    dates = series.index
    values = series.values.reshape(-1, 1)

    # Масштабирование цены
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(values).flatten()

    # Календарные признаки (с сохранением индекса дат)
    dow = pd.Series(dates.dayofweek, index=dates, name='dow')
    month = pd.Series(dates.month - 1, index=dates, name='month')

    dow_ohe = pd.get_dummies(dow, prefix='dow').astype(np.float32)
    month_ohe = pd.get_dummies(month, prefix='m').astype(np.float32)

    # Собираем все признаки по общему индексу
    df_feat = pd.DataFrame({'close': scaled_close}, index=dates)
    df_feat = df_feat.join(dow_ohe).join(month_ohe)   # join по индексу, без сюрпризов

    # Формирование окон
    X, y = [], []
    n = len(df_feat)
    for i in range(lookback, n - 1):   # y – на 1 шаг вперёд
        win = df_feat.iloc[i-lookback:i].values   # lookback строк
        target = df_feat.iloc[i+1]['close']       # цена через 1 день
        X.append(win)
        y.append(target)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler, df_feat

# ---------------------------------------------------------------------
# 2. Модели PyTorch
# ---------------------------------------------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])   # последний временной шаг
        return out.squeeze(-1)

class CNNPredictor(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters,
                               kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size,
                               padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        out = self.fc(x)
        return out.squeeze(-1)

# ---------------------------------------------------------------------
# 3. Обучение с Early Stopping
# ---------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model

# ---------------------------------------------------------------------
# 4. Главная функция-обёртка для walk_forward_validation
# ---------------------------------------------------------------------
def nn_predict(model_type, train_series, steps, lookback=20, epochs=100,
               patience=10, lr=1e-3, batch_size=32, verbose=False):
    """
    Обучает нейросеть (LSTM или CNN) на train_series и рекурсивно прогнозирует
    steps шагов вперёд.

    Параметры:
      model_type: 'lstm' или 'cnn'
      train_series: pd.Series с DatetimeIndex и ценами
      steps: количество будущих точек для предсказания
      lookback: длина окна истории
      epochs, patience, lr, batch_size: параметры обучения

    Возвращает:
      np.array длины steps с прогнозами цен в исходном масштабе.
    """
    # Определяем частоту (нужно для генерации будущих дат)
    freq = infer_frequency(train_series.index)

    # 1. Подготовка данных
    X, y, scaler, df_feat = make_features(train_series, lookback)

    if len(X) < 10:
        # Слишком мало данных – возвращаем наивный прогноз (последнее значение)
        last_val = train_series.iloc[-1]
        return np.full(steps, last_val)

    # Разделение на обучающую и валидационную выборки (по времени)
    val_size = max(1, int(0.2 * len(X)))
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Создание модели
    input_size = X.shape[2]  # количество признаков
    if model_type == 'lstm':
        model = LSTMPredictor(input_size).to(device)
    elif model_type == 'cnn':
        model = CNNPredictor(input_size).to(device)
    else:
        raise ValueError("model_type должен быть 'lstm' или 'cnn'")

    # 3. Обучение
    model = train_model(model, train_loader, val_loader,
                        epochs=epochs, lr=lr, patience=patience)

    # 4. Рекурсивный прогноз на steps шагов
    last_date = train_series.index[-1]
    last_idx = df_feat.index.get_loc(last_date)
    start_idx = last_idx - lookback + 1
    window = df_feat.iloc[start_idx:last_idx+1].values  # (lookback, n_features)
    n_features = window.shape[1]
    feature_cols = df_feat.columns   # сохраняем колонки для прямого обращения

    predictions = []
    current_date = last_date
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            # Генерируем следующую дату
            if freq == 'B':
                current_date = pd.date_range(start=current_date, periods=2, freq='B')[-1]
            else:
                current_date += pd.Timedelta(days=1)

            # Прогноз масштабированной цены
            X_input = torch.from_numpy(window).unsqueeze(0).float().to(device)
            pred_scaled = model(X_input).cpu().item()
            predictions.append(pred_scaled)

            # Создаём вектор признаков для новой даты
            new_row = np.zeros(n_features, dtype=np.float32)

            # close (всегда первый столбец, но надёжнее искать по имени)
            close_idx = feature_cols.get_loc('close')
            new_row[close_idx] = pred_scaled

            # День недели
            dow = current_date.dayofweek   # 0..6
            dow_col = f'dow_{dow}'
            if dow_col in feature_cols:
                new_row[feature_cols.get_loc(dow_col)] = 1.0

            # Месяц
            month = current_date.month - 1  # 0..11
            month_col = f'm_{month}'
            if month_col in feature_cols:
                new_row[feature_cols.get_loc(month_col)] = 1.0

            # Обновляем окно
            window = np.vstack([window[1:], new_row.reshape(1, -1)])

    # 5. Обратное масштабирование
    preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return preds


def make_features_returns(series, lookback):
    """
    Строит окна из доходностей с календарными OHE-признаками.
    Параметры:
        series: pd.Series с доходностями (доли).
        lookback: длина окна.
    Возвращает:
        X: np.array (N, lookback, n_features)
        y: np.array (N,) — следующая доходность
        scaler: MinMaxScaler, подогнанный на доходностях
        df_feat: pd.DataFrame признаков (масштаб. доходность + OHE) с DatetimeIndex
    """
    dates = series.index
    values = series.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_ret = scaler.fit_transform(values).flatten()

    # Календарные признаки
    dow = dates.dayofweek      # 0..6
    month = dates.month - 1    # 0..11

    # Создаём DataFrame с масштабированной доходностью
    df_feat = pd.DataFrame({'return': scaled_ret}, index=dates)

    # Добавляем OHE день недели
    for d in range(7):
        df_feat[f'dow_{d}'] = (dow == d).astype(np.float32)
    # OHE месяц
    for m in range(12):
        df_feat[f'month_{m}'] = (month == m).astype(np.float32)

    # Формируем окна
    X, y = [], []
    n = len(df_feat)
    for i in range(lookback, n - 1):   # y — доходность на i+1 день
        win = df_feat.iloc[i-lookback:i].values
        target = df_feat.iloc[i+1]['return']
        X.append(win)
        y.append(target)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler, df_feat


def nn_predict_returns(model_type, train_series, steps, lookback=20, epochs=100,
                       patience=10, lr=1e-3, batch_size=32, verbose=False):
    """
    Обучает LSTM или CNN на ряде доходностей и рекурсивно прогнозирует steps шагов.
    Возвращает массив доходностей в исходном масштабе (доли).
    """
    # Определяем частоту (нужна для генерации будущих дат)
    freq = infer_frequency(train_series.index)

    # 1. Подготовка данных
    X, y, scaler, df_feat = make_features_returns(train_series, lookback)

    if len(X) < 10:
        return np.full(steps, train_series.iloc[-1])

    # Разделение на train/val
    val_size = max(1, int(0.2 * len(X)))
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Создание модели
    input_size = X.shape[2]  # число признаков
    if model_type == 'lstm':
        model = LSTMPredictor(input_size).to(device)
    elif model_type == 'cnn':
        model = CNNPredictor(input_size).to(device)
    else:
        raise ValueError("model_type должен быть 'lstm' или 'cnn'")

    # 3. Обучение (используем улучшенную train_model с защитой от NaN)
    model = train_model(model, train_loader, val_loader,
                        epochs=epochs, lr=lr, patience=patience)

    # 4. Рекурсивный прогноз на steps шагов
    last_date = train_series.index[-1]
    last_idx = df_feat.index.get_loc(last_date)
    start_idx = last_idx - lookback + 1
    window = df_feat.iloc[start_idx:last_idx+1].values  # (lookback, n_features)
    n_features = window.shape[1]
    feature_cols = df_feat.columns

    predictions = []
    current_date = last_date
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            # Генерируем следующую дату
            if freq == 'B':
                current_date = pd.date_range(start=current_date, periods=2, freq='B')[-1]
            else:
                current_date += pd.Timedelta(days=1)

            # Прогноз масштабированной доходности
            X_input = torch.from_numpy(window).unsqueeze(0).float().to(device)
            pred_scaled = model(X_input).cpu().item()
            predictions.append(pred_scaled)

            # Создаём вектор признаков для новой даты
            new_row = np.zeros(n_features, dtype=np.float32)
            return_idx = feature_cols.get_loc('return')
            new_row[return_idx] = pred_scaled

            # День недели
            dow = current_date.dayofweek
            dow_col = f'dow_{dow}'
            if dow_col in feature_cols:
                new_row[feature_cols.get_loc(dow_col)] = 1.0

            # Месяц
            month = current_date.month - 1
            month_col = f'month_{month}'
            if month_col in feature_cols:
                new_row[feature_cols.get_loc(month_col)] = 1.0

            # Обновляем окно
            window = np.vstack([window[1:], new_row.reshape(1, -1)])

    # 5. Обратное масштабирование
    preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return preds