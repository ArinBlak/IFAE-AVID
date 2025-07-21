# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:42:30 2025

@author: Aries
"""

import os
os.environ["XDG_CONFIG_HOME"] = "/tmp"

import streamlit as st
import pandas as pd
import os
from pyulog import ULog
import matplotlib.pyplot as plt
from pipeline_predict import predict_anomalies

st.set_page_config(page_title="UAV Anomaly Detection", layout="wide")
st.title("🛩️ Real-Time UAV Fault Detection - IFAE Model")

uploaded_file = st.file_uploader("Upload UAV Telemetry CSV or ULog", type=["csv", "ulg"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    
    if file_name.endswith(".csv"):
        st.success("CSV file uploaded...")
        df = pd.read_csv(uploaded_file)
    
    elif file_name.endswith(".ulg"):
        st.success("ULog file uploaded. Parsing... ")
        with open("temp.ulg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        ulog = ULog("temp.ulg")
        
        data_combined = pd.DataFrame(ulog.data_list[0].data)
        data_combined["timestamp"] = ulog.data_list[0].data["timestamp"]
        df = data_combined
        
        os.remove("temp.ulg")

    st.subheader("Raw Uploaded Data")
    st.dataframe(df.head(100))

    with st.spinner("Detecting anomalies..."):
        df_pred = predict_anomalies(df)

    st.success("✅ Anomaly detection complete")

    st.subheader("📉 Anomaly Overview")
    st.write(df_pred['ensemble_anomaly'].value_counts().rename(index={0: 'Normal', 1: 'Anomaly'}))

    st.subheader("📊 Anomaly Timeline")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_pred['ensemble_anomaly'].values, color='red', label="Anomaly")
    ax.set_title("Ensemble Anomaly Detection Over Time")
    ax.set_xlabel("Index")
    ax.set_ylabel("Anomaly (1=Anomaly)")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_pred['ensemble_anomaly'].values, label='Ensemble Anomaly', color='red')
    ax.set_title("Ensemble Anomaly Detection (IF + AE)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Anomaly (1 = Anomaly)")
    ax.grid(True)
    ax.legend()
    
    st.pyplot(fig)

    st.subheader("🧭 PID Error and Actuator Trend (with anomalies)")
    pid_cols = ['roll_error', 'pitch_error', 'yaw_error', 'actuator_total']
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in pid_cols:
        ax.plot(df_pred[col].values, label=col, alpha=0.6)
    anomalies = df_pred[df_pred['ensemble_anomaly'] == 1]
    ax.scatter(anomalies.index, anomalies['actuator_total'], color='red', marker='x', label='Anomaly')
    ax.set_title("PID Components and Detected Faults")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    pid_features = ['roll_error', 'pitch_error', 'yaw_error', 'actuator_total']
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    for feature in pid_features:
        ax2.plot(df_out[feature].values, label=feature, alpha=0.7)
    anomaly_indices = df_out[df_out['ensemble_anomaly'] == 1].index
    ax2.scatter(anomaly_indices, df_out.loc[anomaly_indices, 'actuator_total'],
                color='red', label='Anomaly', marker='x', zorder=5)
    ax2.set_title("PID Feature Trends with Anomalies")
    ax2.set_xlabel("Time Index")
    ax2.set_ylabel("PID Values")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
