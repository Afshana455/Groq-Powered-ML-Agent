import joblib
import numpy as np

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_target(features):
    if isinstance(features, str):
        features = [float(x) for x in features.split(",")]

    features = np.array([features], dtype=float)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return float(prediction[0])
