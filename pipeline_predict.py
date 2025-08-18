from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pyulog import ULog
import tempfile
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.join(os.getcwd(), "models")
iso_scaler = joblib.load(os.path.join(BASE_DIR, "iso_scaler.pkl"))
ae_scaler = joblib.load(os.path.join(BASE_DIR, "ae_scaler.pkl"))
iso_forest = joblib.load(os.path.join(BASE_DIR, "Isolation_forest_model.pkl"))
iso_features = joblib.load(os.path.join(BASE_DIR, "iso_features.pkl"))
ae_model = load_model(os.path.join(BASE_DIR, "autoencoder_model.keras"))
ae_features = joblib.load(os.path.join(BASE_DIR, "ae_features.pkl"))

def predict_anomalies(df):
    Xi = df[iso_features]
    Xi_scaled = iso_scaler.transform(Xi)
    Xa = df[ae_features]
    Xa_scaled = ae_scaler.transform(Xa)

    df["iso_score"] = iso_forest.decision_function(Xi_scaled)
    df["iso_pred"] = iso_forest.predict(Xi_scaled)

    reconstructions = ae_model.predict(Xa_scaled)
    reconstruction_errors = np.mean(np.square(Xa_scaled - reconstructions), axis=1)
    threshold_ae = np.percentile(reconstruction_errors, 95)

    df["ae_score"] = reconstruction_errors
    df["ae_pred"] = (df["ae_score"] > threshold_ae).astype(int)
    df["ensemble_anomaly"] = ((df["iso_pred"] == -1) & (df["ae_pred"] == 1)).astype(int)

    return df

@app.route("/predict", methods=["POST"])
def handle_prediction():
    uploaded_file = request.files["file"]
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = uploaded_file.filename
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(".ulg"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ulg") as temp_file:
                temp_file.write(uploaded_file.read())
                ulog = ULog(temp_file.name)
            df = pd.DataFrame(ulog.data_list[0].data)
            df["timestamp"] = ulog.data_list[0].data["timestamp"]
            os.remove(temp_file.name)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        df_out = predict_anomalies(df)

        return jsonify({
            "head": df_out.head(100).to_dict(orient="records"),
            "anomaly_summary": df_out["ensemble_anomaly"].value_counts().rename(index={0: "Normal", 1: "Anomaly"}).to_dict(),
            "ensemble_anomaly": df_out["ensemble_anomaly"].tolist(),
            "pid_features": {
                col: df_out[col].tolist() for col in ['roll_error', 'pitch_error', 'yaw_error', 'actuator_total'] if col in df_out.columns
            },
            "anomaly_indices": df_out[df_out["ensemble_anomaly"] == 1].index.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
