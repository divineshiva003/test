import numpy as np
import joblib
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("behavior_model.h5")

# Load the same scaler used during training
scaler = joblib.load("scaler.pkl")

# Example 1: Simulated human-like behavior
human_input = np.array([[25, 140, 22, 7500, 8, 0, 18000]])

# Example 2: Simulated bot-like behavior
bot_input = np.array([[200, 20, 5, 800, 1, 1, 3000]])

# Scale inputs just like in training
human_input_scaled = scaler.transform(human_input)
bot_input_scaled = scaler.transform(bot_input)

# Predict probabilities
human_pred = model.predict(human_input_scaled)[0][0]
bot_pred = model.predict(bot_input_scaled)[0][0]

# Output
print(f"Human Prediction Score: {human_pred:.4f} → {'Human' if human_pred > 0.5 else 'Bot'}")
print(f"Bot Prediction Score: {bot_pred:.4f} → {'Human' if bot_pred > 0.5 else 'Bot'}")
