# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:00:02 2025

@author: Aries
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
iso_scaler = joblib.load(os.path.join(BASE_DIR, "iso_scaler.pkl"))
ae_scaler = joblib.load(os.path.join(BASE_DIR, "ae_scaler.pkl"))
iso_forest = joblib.load(os.path.join(BASE_DIR, "Isolation_forest_model.pkl"))
iso_features = joblib.load(os.path.join(BASE_DIR, "iso_features.pkl"))
ae_model = load_model(os.path.join(BASE_DIR, "autoencoder_model.keras"))
ae_features = joblib.load(os.path.join(BASE_DIR, "ae_features.pkl"))

def load_input_data(path):
    df = pd.read_csv("uav_flight_dataset.csv")  
    return df

def predict_anomalies(df):
    Xi = df[iso_features]
    Xi_scaled = iso_scaler.transform(Xi)
    Xi_scaled_df = pd.DataFrame(Xi_scaled, columns=iso_features)
    
    Xa = df[ae_features]
    Xa_scaled = ae_scaler.transform(Xa)
    Xa_scaled_df = pd.DataFrame(Xa_scaled, columns=ae_features)
    
    df["iso_score"] = iso_forest.decision_function(Xi_scaled_df)
    df["iso_pred"] = iso_forest.predict(Xi_scaled_df)
    
    reconstructions = ae_model.predict(Xa_scaled_df)
    reconstruction_errors = np.mean(np.square(Xa_scaled_df - reconstructions), axis=1)
    threshold_ae = np.percentile(reconstruction_errors, 95)
    df["ae_score"] = reconstruction_errors
    df["ae_pred"] = (df["ae_score"] > threshold_ae).astype(int)
    
    df["ensemble_anomaly"] = ((df["iso_pred"] == -1) & (df["ae_pred"] == 1)).astype(int)
    
    return df


