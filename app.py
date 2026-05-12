"""
台股技術分析看板 — Cypher 形態 + 多空追蹤 + 每日目標價
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

from data import TW_STOCKS, PERIODS, fetch_stock_data, get_latest_quote
from analysis import add_all_indicators, detect_cypher, calc_target_price, market_position

st.set_page_config(page_title="台股技術分析看板", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .section-title {
    font-size:1.1rem; font-weight:600;
    border-left:4px solid #636EFA; padding-left:10px; margin:18px 0 8px;
  }
  .badge {
    display:inline-block; padding:4px 14px; border-radius:12px;
    font-size:1rem; font-weight:700;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 設定")
    ticker = st.selectbox("選擇標的", list(TW_STOCKS.keys()),
                          format_func=lambda x: TW_STOCKS[x])
    period = PERIODS[st.selectbox("資料期間", list(PERIODS.keys()), index=2)]
    st.divider()
    show_ma     = st.checkbox("均線 MA",    value=True)
    show_bb     = st.checkbox("布林通道",   value=True)
    show_cypher = st.checkbox("Cypher 形態", value=True)
    show_vol    = st.checkbox("成交量",     value=True)
    st.divider()
    if st.button("🔄 強制更新", use_container_width=True):
        import glob
        for f in glob.glob(".cache/*.json"): os.remove(f)
        st.cache_data.clear(); st.rerun()
    st.caption(f"更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load(ticker, period):
    df = fetch_stock_data(ticker, period)
    return add_all_indicators(df) if df is not None and not df.empty else None

name = TW_STOCKS.get(ticker, ticker)
st.title(f"📈 {name} 技術分析看板")
st.caption("Cypher 諧波形態 ‧ 多空位階 ‧ 每日目標價追蹤")

df = load(ticker, period)
if df is None or df.empty:
    st.error("無法取得資料，請稍後再試。"); st.stop()

quote  = get_latest_quote(ticker)
target = calc_target_price(df)
pos    = market_position(df)

# ── KPI Bar ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
close = quote.get("close",0); chg = quote.get("change",0); pct = quote.get("change_pct",0)
sign  = "+" if chg >= 0 else ""
with c1: st.metric("收盤價", f"{close:,.2f}", f"{sign}{chg:,.2f} ({sign}{pct:.2f}%)")
with c2: st.metric("開盤",   f"{quote.get('open',0):,.2f}")
with c3: st.metric("最高",   f"{quote.get('high',0):,.2f}")
with c4: st.metric("最低",   f"{quote.get('low',0):,.2f}")
with c5:
    vol = quote.get("volume",0)
    st.metric("成交量", f"{vol:,}" if vol < 1e8 else f"{vol/1e8:.2f}億")
with c6: st.metric("多空位階", pos.get("label","—"))
st.divider()

# ── Position + Target ─────────────────────────────────────────────────────────
cl, cr = st.columns(2)
label    = pos.get("label","—")
color    = pos.get("color","#888")
strength = pos.get("strength",0)

with cl:
    st.markdown('<div class="section-title">🎯 市場位階分析</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge" style="background:{color};color:#fff">{label}</span>',
                unsafe_allow_html=True)
    st.progress((strength + 100) // 2)
    st.caption(f"多空強度：{strength:+d} / 100")
    for s in pos.get("signals",[]): st.write(s)

with cr:
    st.markdown('<div class="section-title">📌 明日目標價</div>', unsafe_allow_html=True)
    if target:
        st.caption(f"基準日：{quote.get('date','')}　趨勢：**{target['bias']}**　ATR：{target['atr']:,.2f}")
        t1,t2,t3 = st.columns(3)
        with t1:
            st.metric("🎯 目標價",  f"{target['target_price']:,.2f}")
            st.metric("樞紐 PP",    f"{target['pivot_point']:,.2f}")
        with t2:
            st.metric("壓力 R1",    f"{target['resistance1']:,.2f}")
            st.metric("壓力 R2",    f"{target['resistance2']:,.2f}")
        with t3:
            st.metric("支撐 S1",    f"{target['support1']:,.2f}")
            st.metric("支撐 S2",    f"{target['support2']:,.2f}")
        st.warning(f"止損參考：{target['stop_loss']:,.2f}")
st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 K線技術圖表</div>', unsafe_allow_html=True)
rows = 4 if show_vol else 3
rh   = ([0.45,0.45,0.25,0.20] if show_vol else [0.45,0.45,0.30])[:rows]

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=rh,
                    subplot_titles=(["K線","","RSI(14)","MACD"]+
                                    (["成交量"] if show_vol else []))[:rows])

fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="K線", increasing_line_color="#FF4444", decreasing_line_color="#00C851",
    increasing_fillcolor="#FF4444", decreasing_fillcolor="#00C851"), row=1, col=1)

if show_ma:
    for col,clr,w in [("MA5","#FFD700",1),("MA10","#FF8C00",1),("MA20","#00BFFF",1.5),
                       ("MA60","#FF69B4",1.5),("MA120","#9B59B6",2)]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[col],name=col,
                line=dict(color=clr,width=w),opacity=0.9), row=1,col=1)

if show_bb and "BB_upper" in df.columns:
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_upper"],name="BB上軌",
        line=dict(color="#A9A9A9",dash="dot",width=1),opacity=0.7), row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_lower"],name="BB下軌",
        line=dict(color="#A9A9A9",dash="dot",width=1),opacity=0.7,
        fill="tonexty",fillcolor="rgba(169,169,169,0.07)"), row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_mid"],name="BB中軌",
        line=dict(color="#666",dash="dash",width=1),opacity=0.6), row=1,col=1)

if target:
    x0,x1 = df.index[-12], df.index[-1]
    for val,lbl,clr,dash in [
        (target["pivot_point"],  f"PP {target['pivot_point']:.0f}",   "#888","dot"),
        (target["resistance1"],  f"R1 {target['resistance1']:.0f}",   "#FF6666","dashdot"),
        (target["support1"],     f"S1 {target['support1']:.0f}",      "#66CC66","dashdot"),
        (target["target_price"], f"目標 {target['target_price']:.0f}","#FFD700","dash"),
    ]:
        fig.add_shape(type="line",x0=x0,x1=x1,y0=val,y1=val,
                      line=dict(color=clr,width=1.5,dash=dash),row=1,col=1)
        fig.add_annotation(x=x1,y=val,text=lbl,showarrow=False,
                           font=dict(color=clr,size=10),xanchor="left",row=1,col=1)

if show_cypher:
    for pat in detect_cypher(df):
        pts=[pat.X,pat.A,pat.B,pat.C,pat.D]
        clr = "#00C851" if pat.direction=="bullish" else "#FF4444"
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode="lines+markers+text", name=f"Cypher {pat.direction[:3]} Q{pat.quality:.0f}",
            line=dict(color=clr,width=2,dash="dash"), marker=dict(size=9),
            text=[f"{l}:{p[1]:.0f}" for l,p in zip("XABCD",pts)],
            textposition="top center", textfont=dict(size=8,color=clr)), row=1,col=1)
        fig.add_hrect(y0=pat.D_zone_low,y1=pat.D_zone_high,fillcolor=clr,opacity=0.1,
                      line_width=0,row=1,col=1,
                      annotation_text=f"PRZ {pat.PRZ:.0f}",annotation_font_color=clr)

if "RSI14" in df.columns:
    fig.add_trace(go.Scatter(x=df.index,y=df["RSI14"],name="RSI(14)",
        line=dict(color="#E67E22",width=1.5)), row=2,col=1)
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(255,68,68,0.08)",line_width=0,row=2,col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,200,81,0.08)", line_width=0,row=2,col=1)
    for lv,clr in [(70,"#FF4444"),(50,"#666"),(30,"#00C851")]:
        fig.add_hline(y=lv,line_dash="dot",line_color=clr,line_width=1,row=2,col=1)

if "MACD" in df.columns:
    fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],  name="MACD",  line=dict(color="#3498DB",width=1.5)),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["Signal"],name="Signal",line=dict(color="#E74C3C",width=1.5)),row=3,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df["Hist"],name="Hist",
        marker_color=["#FF4444" if v>=0 else "#00C851" for v in df["Hist"].fillna(0)],
        opacity=0.7), row=3,col=1)

if show_vol:
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="成交量",
        marker_color=["#FF4444" if df["Close"].iloc[i]>=df["Open"].iloc[i] else "#00C851"
                      for i in range(len(df))],opacity=0.7), row=4,col=1)

fig.update_layout(height=820, template="plotly_dark", xaxis_rangeslider_visible=False,
    legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0,font=dict(size=10)),
    margin=dict(l=10,r=65,t=40,b=10),
    paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
fig.update_xaxes(showgrid=True,gridcolor="rgba(255,255,255,0.07)")
fig.update_yaxes(showgrid=True,gridcolor="rgba(255,255,255,0.07)")
st.plotly_chart(fig, use_container_width=True)

# ── Cypher Table ──────────────────────────────────────────────────────────────
if show_cypher:
    st.markdown('<div class="section-title">🔷 Cypher 諧波形態偵測</div>', unsafe_allow_html=True)
    pats = detect_cypher(df)
    if pats:
        st.dataframe(pd.DataFrame([{
            "方向":      "🟢 看多" if p.direction=="bullish" else "🔴 看空",
            "X 點":      f"{p.X[1]:,.2f}",
            "A 點":      f"{p.A[1]:,.2f}",
            "B 點":      f"{p.B[1]:,.2f}",
            "C 點":      f"{p.C[1]:,.2f}",
            "PRZ":       f"{p.PRZ:,.2f}",
            "PRZ 區間":  f"{p.D_zone_low:,.2f} ~ {p.D_zone_high:,.2f}",
            "完成":      "✅" if p.completed else "⏳",
            "品質分":    f"{p.quality:.1f}",
        } for p in pats]), use_container_width=True)
    else:
        st.info("目前區間未偵測到 Cypher 形態，可延長資料期間重試。")

# ── Raw Data ──────────────────────────────────────────────────────────────────
with st.expander("📋 近期數據（含指標）", expanded=False):
    cols = ["Open","High","Low","Close","Volume","MA5","MA20","MA60","RSI14","MACD","ATR14"]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].tail(30).sort_index(ascending=False).copy()
    disp.index = disp.index.strftime("%Y-%m-%d")
    fmt = {c: "{:,.2f}" for c in cols if c != "Volume"}
    fmt["Volume"] = "{:,.0f}"
    st.dataframe(disp.style.format(fmt), use_container_width=True)

# ── Daily Log ─────────────────────────────────────────────────────────────────
LOG = f".cache/log_{ticker.replace('^','').replace('.','_')}.csv"

def load_log():
    if os.path.exists(LOG):
        return pd.read_csv(LOG, parse_dates=["date"])
    return pd.DataFrame(columns=["date","close","target","support","resistance","bias","result"])

def save_log(q, t):
    log   = load_log()
    today = pd.Timestamp(q.get("date", datetime.now().date()))
    if not log.empty and (log["date"] == today).any():
        return log
    log = pd.concat([log, pd.DataFrame([{
        "date": today, "close": q.get("close",0),
        "target": t.get("target_price",0), "support": t.get("support1",0),
        "resistance": t.get("resistance1",0), "bias": t.get("bias",""), "result": "",
    }])], ignore_index=True)
    log.to_csv(LOG, index=False)
    return log

log = save_log(quote, target) if target and quote else load_log()

with st.expander("📅 每日目標價追蹤紀錄", expanded=True):
    if not log.empty:
        disp = log.sort_values("date", ascending=False).head(30).copy()
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
        disp.columns = ["日期","收盤","目標價","支撐S1","壓力R1","偏向","結果"]
        st.dataframe(disp.style.format(
            {"收盤":"{:,.2f}","目標價":"{:,.2f}","支撐S1":"{:,.2f}","壓力R1":"{:,.2f}"}),
            use_container_width=True)
        if len(log) >= 3:
            ls = log.sort_values("date")
            fl = go.Figure()
            fl.add_trace(go.Scatter(x=ls["date"],y=ls["close"],     name="實際收盤",line=dict(color="#00BFFF",width=2)))
            fl.add_trace(go.Scatter(x=ls["date"],y=ls["target"],    name="目標價",  line=dict(color="#FFD700",width=2,dash="dash")))
            fl.add_trace(go.Scatter(x=ls["date"],y=ls["resistance"],name="壓力R1",  line=dict(color="#FF6666",width=1,dash="dot")))
            fl.add_trace(go.Scatter(x=ls["date"],y=ls["support"],   name="支撐S1",  line=dict(color="#66CC66",width=1,dash="dot")))
            fl.update_layout(height=260,template="plotly_dark",
                title="實際收盤 vs 目標價追蹤",margin=dict(l=10,r=10,t=40,b=10),
                paper_bgcolor="#0E1117",plot_bgcolor="#0E1117",
                legend=dict(orientation="h",y=1.15))
            st.plotly_chart(fl, use_container_width=True)
    else:
        st.info("尚無紀錄，每次開啟後自動記錄當日資料。")

st.divider()
st.caption("⚠️ 本看板僅供學習研究，非投資建議。資料來源：Yahoo Finance")
