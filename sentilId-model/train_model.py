import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# 1. Load your behavior dataset
df = pd.read_csv("sentilId-model/behavior_data.csv")

# 2. Separate features and labels
X = df.drop(columns=["label"]).values
y = df["label"].values

# 3. Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for reuse in browser or backend
joblib.dump(scaler, "scaler.pkl")

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Build a small neural network with TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # output: probability of human
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train the model
model.fit(X_train, y_train, epochs=25, batch_size=16, validation_split=0.1)

# 7. Evaluate on test set
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Model trained. Test Accuracy: {acc:.4f}")

# 8. Save the model for conversion to TF.js
model.save("behavior_model.h5")
print("✅ Model saved as behavior_model.h5")
