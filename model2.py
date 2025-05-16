# STEP 1: Install necessary libraries (only if not already installed)
# Uncomment below if using in a fresh Colab environment
# !pip install scikit-learn tensorflow

# STEP 2: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# STEP 3: Create the dataset
data = {
    'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October'],
    '2019': [11.11, 10.91, 9.78, 7.75, 6.15, 7.26, 8.18, 8.01, 7.52, 9.45],
    '2022': [2.16, 2.55, 3.58, 4.11, 4.46, 5.47, 6.69, 5.25, 5.52, 6.77],
    '2023': [8.91, 8.93, 8.25, 6.26, 6.18, 6.68, 7.86, 6.64, 6.67, 8.32],
    '2024': [9.59, 10.03, 8.60, 6.51, 6.00, 7.06, 7.76, 6.36, 6.69, 8.20]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['2019', '2022', '2023']]
y = df['2024']

# STEP 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# A. Sklearn MLPRegressor
# ===============================
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_pred = mlp.predict(X_test_scaled)

mlp_mse = mean_squared_error(y_test, mlp_pred)
mlp_rmse = np.sqrt(mlp_mse)
mlp_r2 = r2_score(y_test, mlp_pred)

print("=== Sklearn MLPRegressor ===")
print(f"MSE: {mlp_mse:.4f}")
print(f"RMSE: {mlp_rmse:.4f}")
print(f"R2: {mlp_r2:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(mlp_pred, label='Predicted', marker='x')
plt.title("MLPRegressor - Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# B. TensorFlow/Keras DNN
# ===============================
model = Sequential([
    Dense(32, activation='relu', input_shape=(3,)),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=200, verbose=0, validation_data=(X_test_scaled, y_test))

# Predict
keras_pred = model.predict(X_test_scaled).flatten()

# Metrics
keras_mse = mean_squared_error(y_test, keras_pred)
keras_rmse = np.sqrt(keras_mse)
keras_r2 = r2_score(y_test, keras_pred)

print("=== Keras Deep Neural Network ===")
print(f"MSE: {keras_mse:.4f}")
print(f"RMSE: {keras_rmse:.4f}")
print(f"R2: {keras_r2:.4f}")

# Plot predictions
plt.figure(figsize=(6, 4))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(keras_pred, label='Predicted', marker='x')
plt.title("Keras DNN - Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
