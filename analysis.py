"""Technical analysis engine: indicators + Cypher harmonic pattern detection."""
import numpy as np
import pandas as pd
from dataclasses import dataclass


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = calc_ema(series, fast)
    ema_slow   = calc_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def calc_bollinger(series: pd.Series, window=20, num_std=2):
    mid   = calc_ma(series, window)
    std   = series.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std

def calc_atr(df: pd.DataFrame, window=14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(window).mean()

def calc_stochastic(df: pd.DataFrame, k_period=14, d_period=3):
    lo = df["Low"].rolling(k_period).min()
    hi = df["High"].rolling(k_period).max()
    k  = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d_period).mean()

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c  = df["Close"]
    df = df.copy()
    df["MA5"]    = calc_ma(c, 5)
    df["MA10"]   = calc_ma(c, 10)
    df["MA20"]   = calc_ma(c, 20)
    df["MA60"]   = calc_ma(c, 60)
    df["MA120"]  = calc_ma(c, 120)
    df["EMA12"]  = calc_ema(c, 12)
    df["EMA26"]  = calc_ema(c, 26)
    df["RSI14"]  = calc_rsi(c, 14)
    df["MACD"], df["Signal"], df["Hist"] = calc_macd(c)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = calc_bollinger(c)
    df["ATR14"]  = calc_atr(df, 14)
    df["StochK"], df["StochD"] = calc_stochastic(df)
    return df


# ── Pivot Points ──────────────────────────────────────────────────────────────

def find_pivots(df: pd.DataFrame, left=3, right=3):
    highs, lows = [], []
    for i in range(left, len(df) - right):
        wh = df["High"].iloc[i - left: i + right + 1]
        wl = df["Low"].iloc[i - left: i + right + 1]
        if df["High"].iloc[i] == wh.max():
            highs.append((df.index[i], float(df["High"].iloc[i])))
        if df["Low"].iloc[i] == wl.min():
            lows.append((df.index[i], float(df["Low"].iloc[i])))
    return highs, lows


# ── Cypher Harmonic Pattern ───────────────────────────────────────────────────

@dataclass
class CypherPattern:
    direction:   str
    X: tuple; A: tuple; B: tuple; C: tuple; D: tuple
    D_zone_low:  float = 0.0
    D_zone_high: float = 0.0
    completed:   bool  = False
    PRZ:         float = 0.0
    quality:     float = 0.0


def _ratio(p1, p2, p3):
    r = p2 - p1
    return 0.0 if abs(r) < 1e-9 else abs((p3 - p1) / r)

def _within(v, lo, hi, tol=0.0):
    return (lo - tol) <= v <= (hi + tol)

def _quality(b, b0, c, c0, d, d0):
    err = (abs(b-b0)/b0 + abs(c-c0)/c0 + abs(d-d0)/d0) / 3
    return max(0, 100 - err * 200)


def detect_cypher(df: pd.DataFrame, tol: float = 0.05) -> list:
    ph, pl = find_pivots(df)
    all_p  = sorted(
        [(d, p, "H") for d, p in ph] + [(d, p, "L") for d, p in pl],
        key=lambda x: x[0]
    )
    patterns = []
    for i in range(len(all_p) - 4):
        Xd,Xp,Xt = all_p[i]
        Ad,Ap,At = all_p[i+1]
        Bd,Bp,Bt = all_p[i+2]
        Cd,Cp,Ct = all_p[i+3]
        Dd,Dp,Dt = all_p[i+4]

        # Bullish: L→H→L→H→(L)
        if Xt=="L" and At=="H" and Bt=="L" and Ct=="H":
            XA = Ap - Xp
            if XA <= 0: continue
            Br = _ratio(Xp, Ap, Bp)
            Ce = (Cp - Xp) / XA
            XC = Cp - Xp
            Dr = _ratio(Xp, Cp, Dp)
            if _within(Br,0.382,0.618,tol) and _within(Ce,1.272,1.414,tol) and _within(Dr,0.732,0.832,tol):
                prz = Xp + XC * 0.782
                q   = _quality(Br,0.5,Ce,1.341,Dr,0.782)
                patterns.append(CypherPattern(
                    "bullish",(Xd,Xp),(Ad,Ap),(Bd,Bp),(Cd,Cp),(Dd,Dp),
                    prz*(1-tol), prz*(1+tol),
                    Dt=="L" and abs(Dp-prz)/prz<0.03,
                    round(prz,2), round(q,1)
                ))

        # Bearish: H→L→H→L→(H)
        if Xt=="H" and At=="L" and Bt=="H" and Ct=="L":
            XA = Xp - Ap
            if XA <= 0: continue
            Br = _ratio(Ap, Xp, Bp)
            Ce = (Xp - Cp) / XA
            XC = Xp - Cp
            Dr = _ratio(Cp, Xp, Dp)
            if _within(Br,0.382,0.618,tol) and _within(Ce,1.272,1.414,tol) and _within(Dr,0.732,0.832,tol):
                prz = Xp - XC * 0.782
                q   = _quality(Br,0.5,Ce,1.341,Dr,0.782)
                patterns.append(CypherPattern(
                    "bearish",(Xd,Xp),(Ad,Ap),(Bd,Bp),(Cd,Cp),(Dd,Dp),
                    prz*(1-tol), prz*(1+tol),
                    Dt=="H" and abs(Dp-prz)/prz<0.03,
                    round(prz,2), round(q,1)
                ))

    patterns.sort(key=lambda p: (p.X[0], p.quality), reverse=True)
    return patterns[:3]


# ── Daily Target Price ─────────────────────────────────────────────────────────

def calc_target_price(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {}
    df   = add_all_indicators(df)
    last = df.iloc[-1]
    close = float(last["Close"])
    atr   = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else close * 0.01
    h, l  = float(last["High"]), float(last["Low"])
    pp = (h + l + close) / 3
    r1 = 2*pp - l;  r2 = pp + (h - l)
    s1 = 2*pp - h;  s2 = pp - (h - l)
    ma20 = float(last["MA20"]) if not np.isnan(last["MA20"]) else close
    ma60 = float(last["MA60"]) if not np.isnan(last["MA60"]) else close
    bias = 1 if close > ma20 > ma60 else (-1 if close < ma20 < ma60 else 0)
    target = round(r1 + atr*0.3*bias, 2) if bias >= 0 else round(s1 + atr*0.3*bias, 2)
    return {
        "pivot_point": round(pp, 2),
        "resistance1": round(r1, 2), "resistance2": round(r2, 2),
        "support1":    round(s1, 2), "support2":    round(s2, 2),
        "target_price": target,
        "stop_loss":   round(s1 if bias >= 0 else r1, 2),
        "atr":         round(atr, 2),
        "bias":        "多頭" if bias > 0 else ("空頭" if bias < 0 else "中性"),
        "rsi":         round(float(last["RSI14"]) if not np.isnan(last["RSI14"]) else 50, 1),
        "macd_hist":   round(float(last["Hist"])  if not np.isnan(last["Hist"])  else 0, 4),
    }


# ── Market Position ────────────────────────────────────────────────────────────

def market_position(df: pd.DataFrame) -> dict:
    if len(df) < 60:
        return {}
    df   = add_all_indicators(df)
    last = df.iloc[-1]
    close = float(last["Close"])
    scores, signals = [], []

    ma5  = float(last.get("MA5",  close))
    ma20 = float(last.get("MA20", close))
    ma60 = float(last.get("MA60", close))
    if   close > ma5 > ma20 > ma60: scores.append(2);  signals.append("均線多頭排列 ✅")
    elif close < ma5 < ma20 < ma60: scores.append(-2); signals.append("均線空頭排列 ❌")
    else:                            scores.append(0);  signals.append("均線盤整中 ⚠️")

    rsi = float(last.get("RSI14", 50))
    if   rsi > 60: scores.append(1);  signals.append(f"RSI強勢 {rsi:.1f} ✅")
    elif rsi < 40: scores.append(-1); signals.append(f"RSI弱勢 {rsi:.1f} ❌")
    else:          scores.append(0);  signals.append(f"RSI中性 {rsi:.1f} ⚠️")

    hist = float(last.get("Hist", 0))
    if hist > 0: scores.append(1);  signals.append("MACD紅柱 ✅")
    else:        scores.append(-1); signals.append("MACD綠柱 ❌")

    bb_mid = float(last.get("BB_mid", close))
    if close > bb_mid: scores.append(1);  signals.append("站上布林中線 ✅")
    else:              scores.append(-1); signals.append("跌破布林中線 ❌")

    total = sum(scores)
    max_s = sum(abs(s) for s in scores)
    strength = int(total / max_s * 100) if max_s else 0
    label, color = (
        ("強勢多頭","#00C851") if total >= 3 else
        ("偏多",    "#76D7C4") if total >= 1 else
        ("強勢空頭","#FF4444") if total <=-3 else
        ("偏空",    "#F1948A") if total <=-1 else
        ("盤整中性","#F39C12")
    )
    return {"label": label, "color": color, "score": total,
            "strength": strength, "signals": signals}
