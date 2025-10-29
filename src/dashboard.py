import os
import time
from datetime import datetime

import requests
import streamlit as st
import plotly.graph_objects as go


API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8001"))
BASE_URL = f"http://{API_HOST}:{API_PORT}"


st.set_page_config(page_title="AI Microgrid Dashboard", page_icon="⚡", layout="wide")


@st.cache_data(ttl=2)
def fetch_dashboard_data():
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/dashboard/metrics", timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except requests.exceptions.RequestException as e:
        return None, f"Service Offline: {e}"


def kpi_section(data):
    col1, col2, col3 = st.columns(3)
    predicted = data.get("predicted_next_hour_kw", [])
    peak_pred = max(predicted) if predicted else 0.0
    with col1:
        st.metric("Predicted Peak Load (Next Hour)", f"{peak_pred:.2f} kW")
    with col2:
        # Simulated KPI
        st.metric("Optimization Commands Sent Today", "12")
    with col3:
        # Simulated KPI
        st.metric("Energy Cost Savings (Simulated)", "$38.20")


def timeseries_chart(data):
    hist = data.get("history", {})
    ts = hist.get("timestamps", [])
    loads = hist.get("load_kw", [])
    predicted = data.get("predicted_next_hour_kw", [])
    threshold = data.get("threshold_kw", 0.0)

    # Convert timestamps to local time strings
    x_hist = [datetime.fromtimestamp(t) for t in ts]
    x_pred = []
    if ts and predicted:
        last_t = ts[-1]
        step = (ts[-1] - ts[-2]) if len(ts) > 1 else 300
        x_pred = [datetime.fromtimestamp(last_t + step * (i + 1)) for i in range(len(predicted))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=loads, mode="lines", name="Historical Load", line=dict(color="#1f77b4")))
    if predicted:
        fig.add_trace(go.Scatter(x=x_pred, y=predicted, mode="lines", name="Predicted Load", line=dict(color="#d62728", dash="dot")))
    if loads:
        fig.add_hline(y=threshold, line=dict(color="#888", width=2), annotation_text="Threshold", annotation_position="top left")

    # Annotation for latest action
    latest_action = data.get("latest_action")
    if latest_action and x_hist:
        fig.add_annotation(
            x=x_hist[-1],
            y=loads[-1],
            text=f"Action: R{latest_action.get('target_relay_id')} -> {latest_action.get('state')}",
            showarrow=True,
            arrowhead=2,
        )
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def relay_and_log(data):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Relays")
        status = data.get("relay_status", {})
        for rid in [1, 2, 3, 4]:
            state = status.get(rid, "off")
            emoji = "🟢" if state == "on" else "🔴"
            st.write(f"Relay {rid}: {emoji} {state.upper()}")
    with c2:
        st.subheader("Optimization Log")
        action = data.get("latest_action")
        if action:
            st.write(f"Latest: {action.get('reason')}")
        else:
            st.write("No actions yet.")


st.title("⚡ AI Microgrid Predictive Load Balancing")
placeholder = st.empty()

while True:
    with placeholder.container():
        data, err = fetch_dashboard_data()
        if err:
            st.warning(err)
        if data:
            kpi_section(data)
            timeseries_chart(data)
            relay_and_log(data)
    time.sleep(5)


