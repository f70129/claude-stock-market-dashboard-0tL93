"""Technical analysis engine: indicators + Cypher harmonic pattern auto-detection."""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_ma(s, w):  return s.rolling(w).mean()
def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()

def calc_rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def calc_macd(s, fast=12, slow=26, sig=9):
    ml = calc_ema(s, fast) - calc_ema(s, slow)
    sl = calc_ema(ml, sig)
    return ml, sl, ml - sl

def calc_bollinger(s, w=20, k=2):
    m = calc_ma(s, w); std = s.rolling(w).std()
    return m + k*std, m, m - k*std

def calc_atr(df, w=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(w).mean()

def calc_stoch(df, k=14, d=3):
    lo = df["Low"].rolling(k).min(); hi = df["High"].rolling(k).max()
    K  = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return K, K.rolling(d).mean()

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]; df = df.copy()
    df["MA5"]   = calc_ma(c, 5);   df["MA10"]  = calc_ma(c, 10)
    df["MA20"]  = calc_ma(c, 20);  df["MA60"]  = calc_ma(c, 60)
    df["MA120"] = calc_ma(c, 120)
    df["RSI14"] = calc_rsi(c, 14)
    df["MACD"], df["Signal"], df["Hist"] = calc_macd(c)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = calc_bollinger(c)
    df["ATR14"] = calc_atr(df, 14)
    df["StochK"], df["StochD"] = calc_stoch(df)
    return df


# ── Pivot Detection ────────────────────────────────────────────────────────────

def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Return (highs, lows) as lists of (index, price)."""
    highs, lows = [], []
    for i in range(left, len(df) - right):
        wh = df["High"].iloc[i-left: i+right+1]
        wl = df["Low"].iloc[i-left:  i+right+1]
        if df["High"].iloc[i] == wh.max():
            highs.append((df.index[i], float(df["High"].iloc[i])))
        if df["Low"].iloc[i] == wl.min():
            lows.append((df.index[i], float(df["Low"].iloc[i])))
    return highs, lows


# ── Cypher Harmonic Pattern ───────────────────────────────────────────────────
#
#  Cypher Fibonacci rules:
#    B retraces XA :  0.382 – 0.618
#    C extends  XA :  1.272 – 1.414  (measured from X)
#    D retraces XC :  0.782          (±tol)
#
# Bullish : X(Lo) → A(Hi) → B(Lo) → C(Hi) → D(Lo/PRZ)
# Bearish : X(Hi) → A(Lo) → B(Hi) → C(Lo) → D(Hi/PRZ)

@dataclass
class CypherPattern:
    direction:   str            # "bullish" | "bearish"
    X: tuple                    # (timestamp, price)
    A: tuple
    B: tuple
    C: tuple
    D: tuple                    # last confirmed pivot near PRZ, or projected
    PRZ:         float = 0.0    # 0.782 retracement of XC
    D_zone_low:  float = 0.0
    D_zone_high: float = 0.0
    completed:   bool  = False  # D pivot confirmed inside PRZ
    quality:     float = 0.0    # 0–100
    B_ratio:     float = 0.0    # actual B retracement of XA
    C_ratio:     float = 0.0    # actual C extension of XA from X
    D_ratio:     float = 0.0    # actual D retracement of XC


def _ret(p1, p2, p3):
    """|(p3–p1)/(p2–p1)|"""
    rng = p2 - p1
    return 0.0 if abs(rng) < 1e-9 else abs((p3 - p1) / rng)

def _ext(x, a, c):
    """(C–X)/(A–X)  for bullish extension"""
    rng = a - x
    return 0.0 if abs(rng) < 1e-9 else (c - x) / rng

def _within(v, lo, hi, tol=0.0):
    return (lo - tol) <= v <= (hi + tol)

def _score(b, c, d):
    b_err = abs(b - 0.500) / 0.500
    c_err = abs(c - 1.341) / 1.341
    d_err = abs(d - 0.782) / 0.782
    return max(0.0, 100 - (b_err + c_err + d_err) / 3 * 200)


def detect_cypher(df: pd.DataFrame,
                  pivot_left:  int   = 3,
                  pivot_right: int   = 3,
                  tol:         float = 0.05) -> list:
    """
    Scan all pivot combos for Cypher harmonic patterns.
    Returns list of CypherPattern sorted by quality desc.
    """
    ph, pl = find_pivots(df, pivot_left, pivot_right)

    all_p = sorted(
        [(d, p, "H") for d, p in ph] + [(d, p, "L") for d, p in pl],
        key=lambda x: x[0]
    )

    results = []
    n = len(all_p)

    for i in range(n - 3):                     # X, A, B, C (D optional)
        Xd,Xp,Xt = all_p[i]
        Ad,Ap,At = all_p[i+1]
        Bd,Bp,Bt = all_p[i+2]
        Cd,Cp,Ct = all_p[i+3]

        # ── Bullish: L H L H ──
        if Xt=="L" and At=="H" and Bt=="L" and Ct=="H":
            XA = Ap - Xp
            if XA <= 0: continue
            Br = _ret(Xp, Ap, Bp)
            Ce = _ext(Xp, Ap, Cp)
            XC = Cp - Xp
            prz = Xp + XC * 0.782

            if (_within(Br, 0.382, 0.618, tol) and
                _within(Ce, 1.272, 1.414, tol)):

                D_candidates = [(d,p,t) for d,p,t in all_p[i+4:]
                                if t=="L" and d > Cd]
                if D_candidates:
                    Dd,Dp,_ = D_candidates[0]
                    Dr = _ret(Xp, Cp, Dp)
                    completed = _within(Dr, 0.732, 0.832, tol)
                else:
                    Dd, Dp, Dr = Cd, prz, 0.782   # projected
                    completed  = False

                q = _score(Br, Ce, Dr if Dr else 0.782)
                results.append(CypherPattern(
                    direction="bullish",
                    X=(Xd,Xp), A=(Ad,Ap), B=(Bd,Bp), C=(Cd,Cp),
                    D=(Dd, Dp),
                    PRZ=round(prz, 4),
                    D_zone_low=round(prz*(1-tol), 4),
                    D_zone_high=round(prz*(1+tol), 4),
                    completed=completed,
                    quality=round(q, 1),
                    B_ratio=round(Br, 4),
                    C_ratio=round(Ce, 4),
                    D_ratio=round(Dr, 4),
                ))

        # ── Bearish: H L H L ──
        if Xt=="H" and At=="L" and Bt=="H" and Ct=="L":
            XA = Xp - Ap
            if XA <= 0: continue
            Br = _ret(Ap, Xp, Bp)
            Ce = _ext(Xp, Ap, Cp)
            Ce = abs((Xp - Cp) / XA) if XA else 0
            XC = Xp - Cp
            prz = Xp - XC * 0.782

            if (_within(Br, 0.382, 0.618, tol) and
                _within(Ce, 1.272, 1.414, tol)):

                D_candidates = [(d,p,t) for d,p,t in all_p[i+4:]
                                if t=="H" and d > Cd]
                if D_candidates:
                    Dd,Dp,_ = D_candidates[0]
                    Dr = _ret(Cp, Xp, Dp)
                    completed = _within(Dr, 0.732, 0.832, tol)
                else:
                    Dd, Dp, Dr = Cd, prz, 0.782
                    completed  = False

                q = _score(Br, Ce, Dr if Dr else 0.782)
                results.append(CypherPattern(
                    direction="bearish",
                    X=(Xd,Xp), A=(Ad,Ap), B=(Bd,Bp), C=(Cd,Cp),
                    D=(Dd, Dp),
                    PRZ=round(prz, 4),
                    D_zone_low=round(prz*(1-tol), 4),
                    D_zone_high=round(prz*(1+tol), 4),
                    completed=completed,
                    quality=round(q, 1),
                    B_ratio=round(Br, 4),
                    C_ratio=round(Ce, 4),
                    D_ratio=round(Dr, 4),
                ))

    # De-duplicate: keep highest quality per X-point
    seen = {}
    for p in results:
        key = (p.X[0], p.direction)
        if key not in seen or p.quality > seen[key].quality:
            seen[key] = p

    return sorted(seen.values(), key=lambda p: p.quality, reverse=True)


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
    r1, r2 = 2*pp - l, pp + (h - l)
    s1, s2 = 2*pp - h, pp - (h - l)
    ma20 = float(last["MA20"]) if not np.isnan(last["MA20"]) else close
    ma60 = float(last["MA60"]) if not np.isnan(last["MA60"]) else close
    bias = 1 if close > ma20 > ma60 else (-1 if close < ma20 < ma60 else 0)
    tgt  = round(r1 + atr*0.3*bias, 2) if bias >= 0 else round(s1 + atr*0.3*bias, 2)
    return {
        "pivot_point": round(pp,2), "resistance1": round(r1,2),
        "resistance2": round(r2,2), "support1": round(s1,2),
        "support2": round(s2,2),    "target_price": tgt,
        "stop_loss": round(s1 if bias>=0 else r1, 2),
        "atr": round(atr,2),
        "bias": "多頭" if bias>0 else ("空頭" if bias<0 else "中性"),
        "rsi":  round(float(last["RSI14"]) if not np.isnan(last["RSI14"]) else 50, 1),
    }


# ── Market Position ────────────────────────────────────────────────────────────

def market_position(df: pd.DataFrame) -> dict:
    if len(df) < 60:
        return {"label":"資料不足","color":"#888","score":0,"strength":0,"signals":[]}
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
    scores.append(1  if hist > 0 else -1)
    signals.append("MACD紅柱 ✅" if hist > 0 else "MACD綠柱 ❌")

    bb_mid = float(last.get("BB_mid", close))
    scores.append(1  if close > bb_mid else -1)
    signals.append("站上布林中線 ✅" if close > bb_mid else "跌破布林中線 ❌")

    total = sum(scores); ms = sum(abs(s) for s in scores)
    strength = int(total/ms*100) if ms else 0
    label, color = (
        ("強勢多頭","#00C851") if total>=3  else
        ("偏多",    "#76D7C4") if total>=1  else
        ("強勢空頭","#FF4444") if total<=-3 else
        ("偏空",    "#F1948A") if total<=-1 else
        ("盤整中性","#F39C12")
    )
    return {"label":label,"color":color,"score":total,"strength":strength,"signals":signals}
