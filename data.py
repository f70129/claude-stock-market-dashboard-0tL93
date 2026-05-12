"""Data fetching module for Taiwan stock market dashboard."""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TW_STOCKS = {
    "^TWII":   "台股加權指數",
    "2330.TW": "台積電 (2330)",
    "2317.TW": "鴻海 (2317)",
    "2454.TW": "聯發科 (2454)",
    "2308.TW": "台達電 (2308)",
    "2882.TW": "國泰金 (2882)",
    "2881.TW": "富邦金 (2881)",
    "2412.TW": "中華電 (2412)",
    "2002.TW": "中鋼 (2002)",
    "1301.TW": "台塑 (1301)",
    "1303.TW": "南亞 (1303)",
    "2886.TW": "兆豐金 (2886)",
    "2891.TW": "中信金 (2891)",
    "3711.TW": "日月光投控 (3711)",
    "2303.TW": "聯電 (2303)",
    "6505.TW": "台塑化 (6505)",
    "2884.TW": "玉山金 (2884)",
    "2357.TW": "華碩 (2357)",
    "2382.TW": "廣達 (2382)",
    "3008.TW": "大立光 (3008)",
    "2395.TW": "研華 (2395)",
    "4938.TW": "和碩 (4938)",
    "2379.TW": "瑞昱 (2379)",
    "2408.TW": "南亞科 (2408)",
    "6669.TW": "緯穎 (6669)",
    "2345.TW": "智邦 (2345)",
    "5880.TW": "合庫金 (5880)",
    "2892.TW": "第一金 (2892)",
    "2880.TW": "華南金 (2880)",
    "2885.TW": "元大金 (2885)",
}

PERIODS = {
    "1個月": "1mo",
    "3個月": "3mo",
    "6個月": "6mo",
    "1年":   "1y",
    "2年":   "2y",
    "3年":   "3y",
    "4年":   "4y",
    "5年":   "5y",
    "6年":   "6y",
    "7年":   "7y",
    "8年":   "8y",
    "9年":   "9y",
    "10年":   "10y",
}


def _cache_path(ticker: str, period: str) -> str:
    safe = ticker.replace("^", "IDX_").replace(".", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{period}.json")


def _is_cache_fresh(path: str, max_age_minutes: int = 30) -> bool:
    if not os.path.exists(path):
        return False
    return (datetime.now().timestamp() - os.path.getmtime(path)) < max_age_minutes * 60


def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    cache_path = _cache_path(ticker, period)
    if _is_cache_fresh(cache_path):
        with open(cache_path) as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        records = df.reset_index().rename(columns={"index": "Date"})
        records["Date"] = records["Date"].astype(str)
        with open(cache_path, "w") as f:
            json.dump(records.to_dict(orient="records"), f)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_multiple(tickers: list, period: str = "6mo") -> dict:
    return {t: fetch_stock_data(t, period) for t in tickers}


def get_latest_quote(ticker: str) -> dict:
    df = fetch_stock_data(ticker, "5d")
    if df is None or df.empty:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    change = last["Close"] - prev["Close"]
    pct = change / prev["Close"] * 100
    return {
        "date":       df.index[-1].strftime("%Y-%m-%d"),
        "open":       round(float(last["Open"]), 2),
        "high":       round(float(last["High"]), 2),
        "low":        round(float(last["Low"]), 2),
        "close":      round(float(last["Close"]), 2),
        "volume":     int(last["Volume"]),
        "change":     round(float(change), 2),
        "change_pct": round(float(pct), 2),
    }
