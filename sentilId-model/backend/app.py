from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model and scaler
model = tf.keras.models.load_model("behavior_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("features")  # Expect list of floats
        if not data:
            return jsonify({"error": "No input features provided"}), 400

        # Preprocess: scale input
        X = np.array(data).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Predict
        prob = model.predict(X_scaled)[0][0]
        prediction = int(prob >= 0.5)

        return jsonify({
            "probability": float(prob),
            "prediction": prediction  # 1 = human, 0 = bot
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
