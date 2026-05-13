"""
Функции загрузки дневных цен закрытия для инструментов:
- MOEX: SBER, YNDX (с учётом смены тикера на YDEX после делистинга)
- Yahoo Finance: BTC-USD, ETH-USD
"""

import os
import time
from datetime import datetime
import pandas as pd
import yfinance as yf
import requests

MOEX_BASE = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"

def fetch_moex_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает всю историю дневных свечей для тикера MOEX.
    Возвращает DataFrame с индексом date и колонкой close.
    """
    print(f"[MOEX] Загрузка {ticker} с {start_date} по {end_date} ...")
    all_rows = []
    start = 0
    columns = None

    while True:
        url = (
            f"{MOEX_BASE}/{ticker}.json"
            f"?from={start_date}&till={end_date}"
            f"&start={start}"
        )
        # Повторы при ошибках сети
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Повтор {attempt+1}/3 после ошибки: {e}")
                time.sleep(5)

        data = resp.json()
        block = data.get("history", {})
        if columns is None:
            columns = block.get("columns", [])
        rows = block.get("data", [])

        if not rows:
            break

        all_rows.extend(rows)
        start += len(rows)

        if len(rows) < 100:  # MOEX отдаёт до 100 строк за запрос
            break

        time.sleep(0.3)

    if not all_rows:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: нет данных для {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=columns)
    # Оставляем нужные колонки: TRADEDATE, CLOSE
    keep_map = {
        "TRADEDATE": "date",
        "CLOSE": "close",
    }
    df = df[[c for c in keep_map if c in df.columns]].rename(columns=keep_map)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    # Удаляем возможные дубликаты дат (оставляем последнее значение)
    df = df[~df.index.duplicated(keep="last")]
    print(f"  Загружено {len(df)} строк")
    return df


def fetch_yahoo_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает дневные данные с Yahoo Finance.
    Возвращает DataFrame с индексом date и колонкой close.
    """
    print(f"[Yahoo] Загрузка {ticker} ...")
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if df.empty:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: нет данных для {ticker}")
        return df

    # Оставляем только close
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    print(f"  Загружено {len(df)} строк")
    return df


def get_russian_stock_data(ticker_main: str, ticker_new: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Объединяет данные основного тикера (например, YNDX) и нового (YDEX),
    если произошла смена тикера. Возвращает единый ряд close.
    """
    df_main = fetch_moex_history(ticker_main, start_date, end_date)
    df_new = fetch_moex_history(ticker_new, start_date, end_date)

    # Если оба не пусты, склеиваем
    if not df_main.empty and not df_new.empty:
        # Берём основной до последней даты, а новый начиная со следующего дня
        last_main_date = df_main.index.max()
        df_new = df_new[df_new.index > last_main_date]
        df_combined = pd.concat([df_main, df_new]).sort_index()
        # Удалим возможные дубли (на всякий случай)
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        print(f"  Объединено: {len(df_main)} + {len(df_new)} = {len(df_combined)} строк")
        return df_combined
    elif not df_main.empty:
        return df_main
    else:
        return df_new
    

def load_data(data_dir="./data"):
    """Загружает 4 подготовленных CSV и возвращает словарь Series."""
    assets = {
        "SBER": "sber_daily.csv",
        "YNDX": "yandex_daily.csv",
        "BTC": "btc_daily.csv",
        "ETH": "eth_daily.csv",
    }
    data = {}
    for name, fname in assets.items():
        path = f"{data_dir}/{fname}"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        # Убедимся, что индекс отсортирован
        df.sort_index(inplace=True)
        data[name] = df["close"].dropna()
    return data

def get_frequency(series):
    """Возвращает 'B' для рядов с пропусками выходных (акции), 'D' для непрерывных (крипто)."""
    diffs = series.index.to_series().diff().dropna().dt.days
    # Если есть дни без данных, то частота бизнес-дней
    if (diffs > 1).any() or (diffs == 3).any():  # понедельник после пятницы
        return 'B'
    else:
        return 'D'