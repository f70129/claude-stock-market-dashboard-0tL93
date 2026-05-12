"""Data fetching module — supports daily / hourly / minute intervals."""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json, os

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

# interval_key -> (yf_interval, yf_period, cache_minutes, label)
INTERVALS = {
    "1分":  ("1m",  "7d",   5,   "1分K"),
    "5分":  ("5m",  "60d",  10,  "5分K"),
    "15分": ("15m", "60d",  15,  "15分K"),
    "30分": ("30m", "60d",  20,  "30分K"),
    "1時":  ("1h",  "180d", 30,  "1時K"),
    "日":   ("1d",  "2y",   60,  "日K"),
    "週":   ("1wk", "5y",   120, "週K"),
}

PERIODS = {
    "7天":  "7d",
    "1個月": "1mo",
    "3個月": "3mo",
    "6個月": "6mo",
    "1年":  "1y",
    "2年":  "2y",
    "5年":  "5y",
}


def _cache_key(ticker, interval_key, custom_period=None):
    safe = ticker.replace("^", "IDX_").replace(".", "_")
    suffix = f"_{custom_period}" if custom_period else ""
    return os.path.join(CACHE_DIR, f"{safe}_{interval_key}{suffix}.json")


def _fresh(path, minutes):
    if not os.path.exists(path):
        return False
    return (datetime.now().timestamp() - os.path.getmtime(path)) < minutes * 60


_DEMO_SEEDS = {
    "^TWII":   (22000, 400, 15_000_000),
    "2330.TW": (850,   25,  25_000_000),
    "2317.TW": (105,    4,  70_000_000),
    "2454.TW": (1100,  45,   8_000_000),
    "2308.TW": (320,   12,  12_000_000),
}
_DEMO_DEFAULT = (100, 5, 3_000_000)

_IV_FREQ = {
    "1分": "min", "5分": "5min", "15分": "15min",
    "30分": "30min", "1時": "h", "日": "B", "週": "W-FRI",
}
_IV_DAYS = {
    "1分": 5, "5分": 55, "15分": 55, "30分": 55,
    "1時": 150, "日": 500, "週": 1200,
}


def _demo(ticker: str, interval_key: str) -> pd.DataFrame:
    base, vol, bvol = _DEMO_SEEDS.get(ticker, _DEMO_DEFAULT)
    freq  = _IV_FREQ.get(interval_key, "B")
    days  = _IV_DAYS.get(interval_key, 500)
    seed  = abs(hash(ticker + interval_key)) % (2**31)
    rng   = np.random.default_rng(seed)
    end   = pd.Timestamp.now().floor("min")
    dates = pd.date_range(end=end, periods=days, freq=freq)
    if interval_key in ("1分","5分","15分","30分","1時"):
        dates = dates[(dates.hour >= 9) & (dates.hour < 14)]
    if len(dates) == 0:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="B")
    ret   = rng.normal(0.0002, vol / base * 0.5, len(dates))
    close = base * np.cumprod(1 + ret)
    high  = close * (1 + rng.uniform(0.001, 0.010, len(dates)))
    low   = close * (1 - rng.uniform(0.001, 0.010, len(dates)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    volume = (rng.lognormal(0, 0.4, len(dates)) * bvol).astype(int)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=dates)


def fetch_stock_data(ticker: str, interval_key: str = "日",
                     custom_period: str = None) -> pd.DataFrame:
    yf_interval, yf_period, cache_min, _ = INTERVALS[interval_key]
    if custom_period:
        yf_period = custom_period
    path = _cache_key(ticker, interval_key, custom_period)

    if _fresh(path, cache_min):
        try:
            with open(path) as f:
                rec = json.load(f)
            df = pd.DataFrame(rec)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            if "Close" in df.columns:
                return df
        except Exception:
            pass

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=yf_period, interval=yf_interval, auto_adjust=True)
        if df.empty:
            return _demo(ticker, interval_key)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        rec = df.reset_index().rename(columns={"index": "Date"})
        rec["Date"] = rec["Date"].astype(str)
        with open(path, "w") as f:
            json.dump(rec.to_dict(orient="records"), f)
        return df
    except Exception:
        return _demo(ticker, interval_key)


def get_latest_quote(ticker: str) -> dict:
    df = fetch_stock_data(ticker, "日")
    if df is None or df.empty:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    change = last["Close"] - prev["Close"]
    pct    = change / prev["Close"] * 100
    return {
        "date":       df.index[-1].strftime("%Y-%m-%d"),
        "open":       round(float(last["Open"]),  2),
        "high":       round(float(last["High"]),  2),
        "low":        round(float(last["Low"]),   2),
        "close":      round(float(last["Close"]), 2),
        "volume":     int(last["Volume"]),
        "change":     round(float(change), 2),
        "change_pct": round(float(pct),    2),
    }
