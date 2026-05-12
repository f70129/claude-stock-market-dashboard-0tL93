"""
台股技術分析看板 — 多時間週期 + Cypher XABCD 自動偵測 + 台指期貨
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

from data import TW_STOCKS, INTERVALS, fetch_stock_data, get_latest_quote
from analysis import add_all_indicators, detect_cypher, calc_target_price, market_position

st.set_page_config(page_title="台股技術分析看板", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .section-title {
    font-size:1.05rem; font-weight:700;
    border-left:4px solid #636EFA; padding-left:10px; margin:14px 0 8px;
  }
  .badge {
    display:inline-block; padding:5px 18px; border-radius:14px;
    font-size:1.05rem; font-weight:700;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 設定")

    ticker = st.selectbox(
        "📌 選擇標的",
        list(TW_STOCKS.keys()),
        format_func=lambda x: TW_STOCKS[x],
    )

    st.markdown("**⏱ 時間週期**")
    interval_key = st.radio(
        "週期", list(INTERVALS.keys()), index=5,
        horizontal=True, label_visibility="collapsed",
    )
    _, _, _, iv_label = INTERVALS[interval_key]

    st.divider()
    st.markdown("**圖表設定**")
    show_ma     = st.checkbox("均線 MA",     value=True)
    show_bb     = st.checkbox("布林通道",    value=True)
    show_cypher = st.checkbox("Cypher 形態", value=True)
    show_vol    = st.checkbox("成交量",      value=True)

    if show_cypher:
        st.markdown("**Cypher 靈敏度**")
        pivot_lr = st.slider("樞紐左右K數", 2, 8, 3)
        fib_tol  = st.slider("Fibonacci 容差 ±%", 1, 10, 5) / 100

    st.divider()
    if st.button("🔄 強制更新資料", use_container_width=True):
        import glob
        for f in glob.glob(".cache/*.json"): os.remove(f)
        st.cache_data.clear(); st.rerun()

    st.caption(f"更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load(ticker, interval_key):
    df = fetch_stock_data(ticker, interval_key)
    if df is not None and not df.empty:
        return add_all_indicators(df)
    return None


name = TW_STOCKS.get(ticker, ticker)
st.title(f"📈 {name}　{iv_label}")
st.caption("Cypher 諧波形態 XABCD 自動偵測 ‧ 多空位階 ‧ 每日目標價")

with st.spinner("載入資料中…"):
    df = load(ticker, interval_key)

if df is None or df.empty:
    st.error("⚠️ 無法取得資料。分鐘資料僅支援近 7 天，請切換其他週期。")
    st.stop()

quote  = get_latest_quote(ticker)
target = calc_target_price(df)
pos    = market_position(df)
cypher_patterns = detect_cypher(df, pivot_lr, pivot_lr, fib_tol) if show_cypher else []


# ── KPI Bar ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
cl  = quote.get("close", float(df["Close"].iloc[-1]))
chg = quote.get("change", 0); pct = quote.get("change_pct", 0)
sign = "+" if chg >= 0 else ""

with c1: st.metric("收盤 / 最新", f"{cl:,.2f}",
                    f"{sign}{chg:,.2f} ({sign}{pct:.2f}%)")
with c2: st.metric("開盤",  f"{quote.get('open',  float(df['Open'].iloc[-1])):,.2f}")
with c3: st.metric("最高",  f"{quote.get('high',  float(df['High'].iloc[-1])):,.2f}")
with c4: st.metric("最低",  f"{quote.get('low',   float(df['Low'].iloc[-1])):,.2f}")
with c5:
    vol = quote.get("volume", int(df["Volume"].iloc[-1]))
    st.metric("成交量", f"{vol:,}" if vol < 1e8 else f"{vol/1e8:.2f}億")
with c6: st.metric("多空位階", pos.get("label","—"))

st.divider()


# ── Position + Target ─────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)
label    = pos.get("label","—")
color    = pos.get("color","#888")
strength = pos.get("strength", 0)

with col_l:
    st.markdown('<div class="section-title">🎯 市場位階</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge" style="background:{color};color:#fff">{label}</span>',
                unsafe_allow_html=True)
    st.progress((strength + 100) // 2)
    st.caption(f"多空強度 {strength:+d} / 100")
    for s in pos.get("signals",[]): st.write(s)

with col_r:
    st.markdown('<div class="section-title">📌 明日目標價</div>', unsafe_allow_html=True)
    if target:
        st.caption(f"基準：{quote.get('date','')}　趨勢：**{target['bias']}**　ATR：{target['atr']:,.2f}")
        t1,t2,t3 = st.columns(3)
        with t1:
            st.metric("🎯 目標",   f"{target['target_price']:,.2f}")
            st.metric("樞紐 PP",   f"{target['pivot_point']:,.2f}")
        with t2:
            st.metric("壓力 R1",   f"{target['resistance1']:,.2f}")
            st.metric("壓力 R2",   f"{target['resistance2']:,.2f}")
        with t3:
            st.metric("支撐 S1",   f"{target['support1']:,.2f}")
            st.metric("支撐 S2",   f"{target['support2']:,.2f}")
        st.warning(f"止損參考：{target['stop_loss']:,.2f}")

st.divider()


# ── Main Chart ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 K線技術圖</div>', unsafe_allow_html=True)

rows = 4 if show_vol else 3
rh   = ([0.50,0.50,0.25,0.18] if show_vol else [0.50,0.50,0.28])[:rows]

fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.025, row_heights=rh,
    subplot_titles=(["K線 + Cypher XABCD","","RSI(14)","MACD"] +
                    (["成交量"] if show_vol else []))[:rows],
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="K線",
    increasing_line_color="#FF4444", decreasing_line_color="#00C851",
    increasing_fillcolor="#FF4444", decreasing_fillcolor="#00C851",
), row=1, col=1)

# MA
if show_ma:
    for cn,clr,w in [("MA5","#FFD700",1),("MA10","#FF8C00",1),
                      ("MA20","#00BFFF",1.5),("MA60","#FF69B4",1.5),("MA120","#9B59B6",2)]:
        if cn in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[cn],name=cn,
                line=dict(color=clr,width=w),opacity=0.9),row=1,col=1)

# Bollinger
if show_bb and "BB_upper" in df.columns:
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_upper"],name="BB上",
        line=dict(color="#448",dash="dot",width=1),opacity=0.7),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_lower"],name="BB下",
        line=dict(color="#448",dash="dot",width=1),opacity=0.7,
        fill="tonexty",fillcolor="rgba(100,100,180,0.06)"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_mid"],name="BB中",
        line=dict(color="#558",dash="dash",width=1),opacity=0.55),row=1,col=1)

# Pivot Lines
if target:
    x0,x1 = df.index[max(-20,-len(df))], df.index[-1]
    for val,lbl,clr,dash in [
        (target["pivot_point"],  f"PP {target['pivot_point']:,.0f}",   "#888","dot"),
        (target["resistance1"],  f"R1 {target['resistance1']:,.0f}",   "#FF6666","dashdot"),
        (target["support1"],     f"S1 {target['support1']:,.0f}",      "#66CC66","dashdot"),
        (target["target_price"], f"目標 {target['target_price']:,.0f}","#FFD700","dash"),
    ]:
        fig.add_shape(type="line",x0=x0,x1=x1,y0=val,y1=val,
                      line=dict(color=clr,width=1.4,dash=dash),row=1,col=1)
        fig.add_annotation(x=x1,y=val,text=lbl,showarrow=False,
                           font=dict(color=clr,size=9.5),
                           xanchor="left",yanchor="middle",row=1,col=1)

# ── Cypher XABCD ──────────────────────────────────────────────────────────────
PLBLS = ["X","A","B","C","D"]
for pat in cypher_patterns:
    pts = [pat.X, pat.A, pat.B, pat.C, pat.D]
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    clr = "#00FF88" if pat.direction == "bullish" else "#FF4466"

    # XABCD line
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers+text",
        name=f"Cypher {'多' if pat.direction=='bullish' else '空'} Q{pat.quality:.0f}",
        line=dict(color=clr, width=2, dash="dash"),
        marker=dict(size=11, color=clr, symbol="circle",
                    line=dict(color="#fff", width=1.5)),
        text=[f"<b>{lb}</b><br>{p[1]:,.1f}" for lb,p in zip(PLBLS, pts)],
        textposition=[
            "bottom center","top center","bottom center","top center","bottom center"
        ] if pat.direction=="bullish" else [
            "top center","bottom center","top center","bottom center","top center"
        ],
        textfont=dict(size=9, color=clr),
    ), row=1, col=1)

    # PRZ zone
    fig.add_hrect(
        y0=pat.D_zone_low, y1=pat.D_zone_high,
        fillcolor=clr, opacity=0.10, line_width=0,
        row=1, col=1,
        annotation_text=f"PRZ {pat.PRZ:,.1f}",
        annotation_font=dict(color=clr, size=9),
        annotation_position="right",
    )

# RSI
if "RSI14" in df.columns:
    fig.add_trace(go.Scatter(x=df.index,y=df["RSI14"],name="RSI(14)",
        line=dict(color="#E67E22",width=1.4)),row=2,col=1)
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(255,68,68,0.07)",line_width=0,row=2,col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,200,81,0.07)", line_width=0,row=2,col=1)
    for lv,clr in [(70,"#FF4444"),(50,"#555"),(30,"#00C851")]:
        fig.add_hline(y=lv,line_dash="dot",line_color=clr,line_width=1,row=2,col=1)
    rsi_now = float(df["RSI14"].iloc[-1])
    fig.add_annotation(x=df.index[-1],y=rsi_now,
        text=f" {rsi_now:.1f}",showarrow=False,
        font=dict(color="#E67E22",size=9),xanchor="left",row=2,col=1)

# MACD
if "MACD" in df.columns:
    fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],  name="MACD",
        line=dict(color="#3498DB",width=1.4)),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["Signal"],name="Signal",
        line=dict(color="#E74C3C",width=1.4)),row=3,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df["Hist"],name="Hist",
        marker_color=["#FF4444" if v>=0 else "#00C851"
                      for v in df["Hist"].fillna(0)],opacity=0.7),row=3,col=1)
    fig.add_hline(y=0,line_color="#444",line_width=0.8,row=3,col=1)

# Volume
if show_vol:
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="成交量",
        marker_color=["#FF4444" if df["Close"].iloc[i]>=df["Open"].iloc[i]
                      else "#00C851" for i in range(len(df))],
        opacity=0.65),row=4,col=1)

fig.update_layout(
    height=860, template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0,
                font=dict(size=9.5),bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10,r=75,t=45,b=10),
    paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
st.plotly_chart(fig, use_container_width=True)


# ── Cypher Detail ─────────────────────────────────────────────────────────────
if show_cypher:
    st.markdown('<div class="section-title">🔷 Cypher XABCD 形態詳情</div>',
                unsafe_allow_html=True)
    if cypher_patterns:
        def fmt_pt(pt):
            ts = pt[0]
            fmt = "%m/%d %H:%M" if hasattr(ts,"hour") and interval_key not in ("日","週") else "%m/%d"
            return f"{pt[1]:,.2f}　({ts.strftime(fmt) if hasattr(ts,'strftime') else str(ts)[:10]})"

        st.dataframe(pd.DataFrame([{
            "方向":     "🟢 看多" if p.direction=="bullish" else "🔴 看空",
            "X":        fmt_pt(p.X),
            "A":        fmt_pt(p.A),
            "B":        fmt_pt(p.B),
            "C":        fmt_pt(p.C),
            "PRZ (D目標)": f"{p.PRZ:,.2f}",
            "PRZ 區間": f"{p.D_zone_low:,.2f} ～ {p.D_zone_high:,.2f}",
            "B fib":    f"{p.B_ratio:.3f}  (0.382~0.618)",
            "C fib":    f"{p.C_ratio:.3f}  (1.272~1.414)",
            "D fib":    f"{p.D_ratio:.3f}  (≈ 0.782)",
            "D完成":    "✅" if p.completed else "⏳",
            "品質":     f"{p.quality:.1f} / 100",
        } for p in cypher_patterns]), use_container_width=True)

        # Fibonacci 比例燈號
        st.markdown("**各點 Fibonacci 比例對照**")
        fcols = st.columns(min(len(cypher_patterns), 4))
        for col_f, pat in zip(fcols, cypher_patterns):
            with col_f:
                d_lbl = "🟢 看多" if pat.direction=="bullish" else "🔴 看空"
                st.caption(f"{d_lbl}　品質 {pat.quality:.0f}/100")
                for lbl, actual, lo, hi in [
                    ("B", pat.B_ratio, 0.382, 0.618),
                    ("C", pat.C_ratio, 1.272, 1.414),
                    ("D", pat.D_ratio, 0.732, 0.832),
                ]:
                    ok = lo <= actual <= hi
                    st.write(f"{'✅' if ok else '⚠️'} **{lbl}** `{actual:.3f}` [{lo}~{hi}]")
    else:
        st.info(f"目前 {iv_label} 週期未偵測到 Cypher 形態。"
                "可嘗試：降低左右K數、增大容差，或換較長週期。")

    with st.expander("📖 Cypher 形態說明"):
        st.markdown("""
| 點位 | Fibonacci 範圍 | 說明 |
|------|--------------|------|
| X→A | — | 起始驅動波 |
| A→B | **0.382~0.618** XA | B 回測 XA |
| X→C | **1.272~1.414** XA | C 延伸突破 A（以 X 為原點） |
| X→D | **0.782** XC | D = PRZ 潛在反轉區 |

🟢 **看多**：D 跌入 PRZ → 做多，止損破 X  
🔴 **看空**：D 漲入 PRZ → 做空，止損破 X
        """)


# ── Raw Data ──────────────────────────────────────────────────────────────────
with st.expander("📋 近期K線數據（含指標）"):
    show_cols = ["Open","High","Low","Close","Volume",
                 "MA5","MA20","MA60","RSI14","MACD","Signal","ATR14"]
    avail = [c for c in show_cols if c in df.columns]
    disp  = df[avail].tail(50).sort_index(ascending=False).copy()
    fmt   = {c:"{:,.2f}" for c in avail if c!="Volume"}; fmt["Volume"]="{:,.0f}"
    ts_fmt = "%Y-%m-%d %H:%M" if interval_key not in ("日","週") else "%Y-%m-%d"
    disp.index = disp.index.strftime(ts_fmt)
    st.dataframe(disp.style.format(fmt), use_container_width=True)


# ── Daily Log ─────────────────────────────────────────────────────────────────
LOG = f".cache/log_{ticker.replace('^','').replace('=','_').replace('.','_')}_{interval_key}.csv"

def load_log():
    if os.path.exists(LOG):
        return pd.read_csv(LOG, parse_dates=["date"])
    return pd.DataFrame(columns=["date","close","target","support","resistance","bias","result"])

def save_log(q, t):
    log   = load_log()
    today = pd.Timestamp(q.get("date", datetime.now().date()))
    if not log.empty and (log["date"]==today).any(): return log
    return pd.concat([log, pd.DataFrame([{
        "date": today, "close": q.get("close",0),
        "target": t.get("target_price",0), "support": t.get("support1",0),
        "resistance": t.get("resistance1",0), "bias": t.get("bias",""), "result":"",
    }])], ignore_index=True).pipe(lambda d: (d.to_csv(LOG,index=False), d)[1])

log = save_log(quote, target) if target and quote else load_log()

with st.expander("📅 每日目標價追蹤紀錄", expanded=True):
    if not log.empty:
        disp = log.sort_values("date",ascending=False).head(30).copy()
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
        disp.columns = ["日期","收盤","目標價","支撐S1","壓力R1","偏向","結果"]
        st.dataframe(disp.style.format(
            {"收盤":"{:,.2f}","目標價":"{:,.2f}","支撐S1":"{:,.2f}","壓力R1":"{:,.2f}"}),
            use_container_width=True)
        if len(log) >= 3:
            ls = log.sort_values("date")
            fl = go.Figure()
            for y,nm,clr,dash in [
                (ls["close"],      "實際收盤","#00BFFF","solid"),
                (ls["target"],     "目標價",  "#FFD700","dash"),
                (ls["resistance"], "壓力R1",  "#FF6666","dot"),
                (ls["support"],    "支撐S1",  "#66CC66","dot"),
            ]:
                fl.add_trace(go.Scatter(x=ls["date"],y=y,name=nm,
                    line=dict(color=clr,width=2 if dash=="solid" else 1,dash=dash)))
            fl.update_layout(height=240,template="plotly_dark",
                title="收盤 vs 目標價追蹤",margin=dict(l=10,r=10,t=36,b=10),
                paper_bgcolor="#0E1117",plot_bgcolor="#0E1117",
                legend=dict(orientation="h",y=1.2))
            st.plotly_chart(fl, use_container_width=True)
    else:
        st.info("尚無紀錄，開啟後自動記錄當日資料。")

st.divider()
st.caption("⚠️ 本看板僅供學習研究，非投資建議。資料來源：Yahoo Finance  "
           f"｜  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
