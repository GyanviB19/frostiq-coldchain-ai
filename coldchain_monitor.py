
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="ColdChain AI Monitor", layout="wide")
st.title("📦 FrostIQ ColdChain – Real-Time Anomaly Detection")

# Define sensor simulation function
def generate_sensor_data():
    temp = round(np.random.uniform(2, 12), 2)
    if np.random.rand() < 0.05:  # Inject anomaly
        temp = round(np.random.uniform(15, 20), 2)
    return {
        "timestamp": datetime.now(),
        "temperature": temp,
        "humidity": round(np.random.uniform(60, 95), 2),
        "door_status": np.random.choice(["open", "closed"])
    }

# Preload with 5 entries if empty
if 'data' not in st.session_state or st.session_state.data.empty:
    st.session_state.data = pd.DataFrame([generate_sensor_data() for _ in range(5)])

# Button to simulate a new reading
if st.button("➕ Simulate New Reading"):
    new_data = generate_sensor_data()
    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_data])], ignore_index=True)
    st.success("New reading added!")

# Prepare data
df = st.session_state.data.copy()
df["door_status_encoded"] = df["door_status"].map({"closed": 0, "open": 1})
features = df[["temperature", "humidity", "door_status_encoded"]]

# Anomaly detection
if len(features) > 5:
    model = IsolationForest(contamination=0.1)
    df["anomaly"] = model.fit_predict(features)
    df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})
else:
    df["anomaly"] = 1
    df["anomaly_label"] = "Normal"

# Display metrics
anomalies = df[df["anomaly_label"] == "Anomaly"]
st.metric("Total Readings", len(df))
st.metric("Detected Anomalies", len(anomalies))

# Plot temperature with anomalies
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["timestamp"], df["temperature"], label="Temperature", color='blue')
ax.scatter(anomalies["timestamp"], anomalies["temperature"], color="red", label="Anomaly", marker="x")
ax.set_title("Temperature over Time with Anomalies")
ax.set_ylabel("°C")
ax.set_xlabel("Timestamp")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.info("ℹ️ Click 'Simulate New Reading' to keep adding data.")

