import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime
import joblib
import os


# ══════════════════════════════════════════════════════════════
#  PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════
def predict_next_12h(df):
    model = load_model("lstm_multivariate_model.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
    features = ["tempm", "rain", "hum", "wspdm"]
    SEQ_LEN = 48

    input_data = df[features].tail(SEQ_LEN).values.astype(float)
    scaled = scaler.transform(input_data)
    X = scaled[np.newaxis, ...]

    pred_scaled = model.predict(X, verbose=0)[0]
    pred_actual = scaler.inverse_transform(pred_scaled)

    last_time = df.index[-1]
    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=12,
        freq="h"
    )
    return future_times, pred_actual


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Delhi Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)


# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("delhi_weather_datasets.csv")
    df.columns = df.columns.str.strip()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df = df.dropna(subset=["datetime_utc"])
    df.set_index("datetime_utc", inplace=True)
    df.sort_index(inplace=True)
    for col in ["tempm", "rain", "hum", "wspdm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.ffill().bfill()
    return df


df = load_data()


# ══════════════════════════════════════════════════════════════
#  SAFETY CHECKS
# ══════════════════════════════════════════════════════════════
if not os.path.exists("scaler.pkl"):
    st.error("❌ scaler.pkl not found! Please run `python train_lstm.py` first.")
    st.stop()

if not os.path.exists("lstm_multivariate_model.keras"):
    st.error("❌ Model not found! Please run `python train_lstm.py` first.")
    st.stop()


# ══════════════════════════════════════════════════════════════
#  RUN PREDICTION
# ══════════════════════════════════════════════════════════════
with st.spinner("🔮 Running LSTM prediction..."):
    future_times, pred = predict_next_12h(df)

pred_temp = pred[:, 0]
pred_rain = pred[:, 1]
pred_hum  = pred[:, 2]
pred_wind = pred[:, 3]

df_ctx       = df.tail(48)
current_temp = float(df["tempm"].iloc[-1])
current_hum  = float(df["hum"].iloc[-1])
current_wind = float(df["wspdm"].iloc[-1])


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.title("🌤️ Delhi Weather — Next 12-Hour LSTM Forecast")
st.caption(
    f"📅 Last data point: **{df.index[-1].strftime('%d %b %Y, %H:%M')}**"
    f"  |  🕐 Generated: **{datetime.now().strftime('%d %b %Y, %H:%M')}**"
)
st.divider()


# ══════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌡️ Current Temp",  f"{current_temp:.1f} °C")
c2.metric("🔺 Forecast High", f"{pred_temp.max():.1f} °C",
          f"{pred_temp.max() - current_temp:+.1f} °C")
c3.metric("💧 Avg Humidity",  f"{pred_hum.mean():.1f} %",
          f"{pred_hum.mean() - current_hum:+.1f} %")
c4.metric("💨 Avg Wind",      f"{max(pred_wind.mean(), 0):.1f} km/h",
          f"{pred_wind.mean() - current_wind:+.1f} km/h")
st.divider()


# ══════════════════════════════════════════════════════════════
#  TEMPERATURE CHART
# ══════════════════════════════════════════════════════════════
st.subheader("🌡️ Temperature Forecast (°C)")

fig_temp = go.Figure()

fig_temp.add_trace(go.Scatter(
    x=df_ctx.index,
    y=df_ctx["tempm"],
    name="Historical",
    line=dict(color="#4A90D9", width=2),
    mode="lines"
))

fig_temp.add_trace(go.Scatter(
    x=future_times,
    y=pred_temp,
    name="LSTM Forecast",
    line=dict(color="#FF6B35", width=3, dash="dot"),
    mode="lines+markers",
    marker=dict(size=8)
))

sigma = 1.5
fig_temp.add_trace(go.Scatter(
    x=list(future_times) + list(future_times[::-1]),
    y=list(pred_temp + sigma) + list((pred_temp - sigma)[::-1]),
    fill="toself",
    fillcolor="rgba(255,107,53,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="± 1.5°C Band",
    showlegend=True
))

fig_temp.update_layout(
    height=420,
    hovermode="x unified",
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white",
    xaxis=dict(gridcolor="#2a2a2a", title="Time"),
    yaxis=dict(gridcolor="#2a2a2a", title="Temperature (°C)"),
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig_temp, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  HUMIDITY + WIND SIDE BY SIDE
# ══════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.subheader("💧 Humidity Forecast (%)")
    fig_hum = go.Figure()
    fig_hum.add_trace(go.Scatter(
        x=df_ctx.index,
        y=df_ctx["hum"],
        name="Historical",
        line=dict(color="#4ECDC4", width=2),
        mode="lines"
    ))
    fig_hum.add_trace(go.Scatter(
        x=future_times,
        y=pred_hum,
        name="Forecast",
        line=dict(color="#FF6B35", width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=6)
    ))
    fig_hum.update_layout(
        height=320,
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a", title="%"),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_hum, use_container_width=True)

with col2:
    st.subheader("💨 Wind Speed Forecast (km/h)")
    fig_wind = go.Figure()
    fig_wind.add_trace(go.Scatter(
        x=df_ctx.index,
        y=df_ctx["wspdm"],
        name="Historical",
        line=dict(color="#45B7D1", width=2),
        mode="lines"
    ))
    fig_wind.add_trace(go.Scatter(
        x=future_times,
        y=np.maximum(pred_wind, 0),
        name="Forecast",
        line=dict(color="#FF6B35", width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=6)
    ))
    fig_wind.update_layout(
        height=320,
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a", title="km/h"),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_wind, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  RAIN CHART
# ══════════════════════════════════════════════════════════════
st.subheader("🌧️ Rain Forecast (mm)")

fig_rain = go.Figure()
fig_rain.add_trace(go.Bar(
    x=df_ctx.index,
    y=df_ctx["rain"],
    name="Historical Rain",
    marker_color="#4A90D9",
    opacity=0.6
))
fig_rain.add_trace(go.Bar(
    x=future_times,
    y=np.maximum(pred_rain, 0),
    name="Forecast Rain",
    marker_color="#FF6B35",
    opacity=0.85
))
fig_rain.update_layout(
    height=300,
    barmode="overlay",
    hovermode="x unified",
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white",
    xaxis=dict(gridcolor="#2a2a2a", title="Time"),
    yaxis=dict(gridcolor="#2a2a2a", title="Rain (mm)"),
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig_rain, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  FORECAST TABLE
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("📋 Hourly Forecast Table")

prev_temp = current_temp
rows = []
for i in range(12):
    delta = float(pred_temp[i]) - prev_temp
    arrow = "🔺" if delta > 0 else "🔻"
    rows.append({
        "Hour":         future_times[i].strftime("%d %b %Y  %H:%M"),
        "Temp (°C)":    f"{pred_temp[i]:.1f}",
        "Trend":        f"{arrow} {abs(delta):.1f}°C",
        "Rain (mm)":    f"{max(pred_rain[i], 0):.2f}",
        "Humidity (%)": f"{pred_hum[i]:.1f}",
        "Wind (km/h)":  f"{max(pred_wind[i], 0):.1f}",
    })
    prev_temp = float(pred_temp[i])

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Dashboard Info")
    st.info(
        "**Model:** 2-Layer LSTM\n\n"
        "**Lookback:** 48 hours\n\n"
        "**Forecast:** 12 hours\n\n"
        "**Features:** Temp · Rain · Humidity · Wind\n\n"
        "**Dataset:** Delhi Weather"
    )
    st.divider()
    st.subheader("📊 Dataset Stats")
    st.write(f"**Total rows:** {len(df):,}")
    st.write(f"**From:** {df.index[0].strftime('%d %b %Y')}")
    st.write(f"**To:**   {df.index[-1].strftime('%d %b %Y')}")
    st.divider()
    show_raw = st.checkbox("🗂️ Show Raw Data (last 48 rows)", value=False)

if show_raw:
    st.divider()
    st.subheader("🗂️ Raw Data — Last 48 Hours")
    st.dataframe(
        df[["tempm", "rain", "hum", "wspdm"]].tail(48).rename(columns={
            "tempm": "Temp (°C)",
            "rain":  "Rain (mm)",
            "hum":   "Humidity (%)",
            "wspdm": "Wind (km/h)"
        }),
        use_container_width=True
    )

st.caption("Built with LSTM · TensorFlow · Streamlit · Plotly")
