
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="ColdChain AI – Full Dataset Monitor", layout="wide")
st.title("📦 FrostIQ ColdChain – Full Dataset Monitoring & Anomaly Detection")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("coldchain_full_dataset.csv", parse_dates=["timestamp"])

df = load_data()
df["door_status_encoded"] = df["door_status"].map({"closed": 0, "open": 1})
features = df[["temperature", "humidity", "external_temp", "wind_speed", "door_status_encoded"]]

# Anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(features)
df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

# Display metrics
st.metric("Total Records", len(df))
st.metric("Detected Anomalies", len(df[df["anomaly_label"] == "Anomaly"]))

# Line chart of temperature
st.subheader("Temperature Over Time")
fig_temp, ax_temp = plt.subplots(figsize=(12, 4))
ax_temp.plot(df["timestamp"], df["temperature"], label="Temperature", color='blue')
anomaly_pts = df[df["anomaly_label"] == "Anomaly"]
ax_temp.scatter(anomaly_pts["timestamp"], anomaly_pts["temperature"], color="red", label="Anomaly", marker="x")
ax_temp.set_xlabel("Time")
ax_temp.set_ylabel("°C")
ax_temp.set_title("Internal Temperature Readings with Anomalies")
ax_temp.legend()
st.pyplot(fig_temp)

# Weather correlation
st.subheader("Weather Correlation")
fig_weather = px.scatter(df, x="external_temp", y="temperature", color="anomaly_label",
                         labels={"external_temp": "External Temp (°C)", "temperature": "Internal Temp (°C)"},
                         title="Internal vs External Temperature (Colored by Anomaly)")
st.plotly_chart(fig_weather)

# GPS map of readings
st.subheader("GPS Route of All Readings")
fig_map = px.scatter_mapbox(df, lat="gps_lat", lon="gps_lon", color="anomaly_label",
                            zoom=10, height=400,
                            mapbox_style="open-street-map",
                            title="ColdChain GPS Route with Anomalies Highlighted")
st.plotly_chart(fig_map)

st.caption("Data Source: Simulated IoT & Weather for ColdChain logistics.")
