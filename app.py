

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Correct template + static mapping for Render
app = Flask(__name__, template_folder="templates", static_folder="static")

# ----------------------------
# MODEL LOADING (Render ready)
# ----------------------------

# On Render, model will be stored on Render Disk:
MODEL_PATH = "model_compressed.pkl"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("Model file not found on Render Disk.")
    model = None

# ---------------------------------------
# Feature Configuration
# ---------------------------------------
MODEL_FEATURES = [
    'Weight', 'Height', 'Green_Vegetables', 'General_Health',
    'Fruit', 'Fried_Potato', 'BMI', 'Age', 'Alcohol'
]

SCALERS = {
    'Weight': {'min': 30, 'max': 150},
    'Height': {'min': 120, 'max': 220},
    'BMI': {'min': 15, 'max': 50},
    'Age': {'min': 18, 'max': 80},
    'Fruit': {'min': 0, 'max': 30},
    'Green_Vegetables': {'min': 0, 'max': 30},
    'Fried_Potato': {'min': 0, 'max': 30},
    'Alcohol': {'min': 0, 'max': 30},
    'General_Health': {
        'Poor': 0.0, 'Fair': 0.25, 'Good': 0.5, 'Very_Good': 0.75, 'Excellent': 1.0
    }
}


def normalize(value, feature_name):
    if feature_name in SCALERS:
        config = SCALERS[feature_name]

        if isinstance(config, dict) and 'min' in config:
            try:
                val = float(value)
                return (val - config['min']) / (config['max'] - config['min'])
            except:
                return 0.0

        elif isinstance(config, dict):
            return config.get(value, 0.0)

    try:
        return float(value)
    except:
        return 0.0

@app.route("/")
def home():
    return render_template("home.html")
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/calculate")
def calculate():
    return render_template("index.html")

@app.route("/recommendation/<int:level>")
def recommendation(level):
    return render_template("recommendation.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json

    input_vector = []
    for feature in MODEL_FEATURES:
        raw_val = data.get(feature)
        norm_val = normalize(raw_val, feature)
        norm_val = max(0.0, min(1.0, norm_val))
        # input_vector.append(norm_val)
        input_vector.append(max(0.0, min(1.0, norm_val)))

    try:
        prediction_prob = model.predict_proba([input_vector])[0][1]
        prediction_class = int(model.predict([input_vector])[0])

        return jsonify({
            "probability": float(prediction_prob),
            "class": prediction_class,
            "risk_level": "High" if prediction_prob > 0.5 else "Low"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# RENDER DEPLOYMENT ENTRY POINT
# ---------------------------------------
if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 10000))
    # app.run(host="0.0.0.0", port=port)
    app.run(host="0.0.0.0", port=10000, debug=True)