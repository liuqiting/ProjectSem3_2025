import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load data and scaler
data = pd.read_csv('preprocessed_data.csv')
scaler = joblib.load('scaler.pkl')

# Create sequences for LSTM (input shape: [samples, time_steps, features])
seq_len = 24
n_features = 1  # Use humidity as the only feature
X, y = [], []
for i in range(len(data) - seq_len):
    X.append(data['humidity_norm'].iloc[i:i+seq_len].values.reshape(-1, 1))
    y.append(data['humidity_norm'].iloc[i+seq_len])
X = np.array(X)
y = np.array(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build lightweight LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(seq_len, n_features)))  # 32 hidden units (lightweight design)
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=8,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
y_pred = model.predict(X_test)

# Inverse normalization
y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Model evaluation
mse = mean_squared_error(y_test_denorm, y_pred_denorm)
mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
print(f"LSTM Model Evaluation:")
print(f"- Mean Squared Error (MSE): {mse:.2f}")
print(f"- Mean Absolute Error (MAE): {mae:.2f}")

# Save model and results
model.save('lstm_model.h5')
lstm_eval = pd.DataFrame({
    'True_Humidity': y_test_denorm,
    'LSTM_Predicted_Humidity': y_pred_denorm,
    'Error': y_test_denorm - y_pred_denorm
})
lstm_eval.to_csv('lstm_model_evaluation.csv', index=False)

# Plot prediction results
plt.figure(figsize=(10, 4))
plt.plot(y_test_denorm, label='True Humidity', color='blue')
plt.plot(y_pred_denorm, label='LSTM Predicted Humidity', color='red', linestyle=':')
plt.xlabel('Test Sample Index')
plt.ylabel('Humidity (%)')
plt.title('LSTM Model: Periodic Humidity Prediction Results')
plt.legend()
plt.savefig('lstm_prediction_results.png', dpi=300)
plt.close()

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Model Training Loss Trend')
plt.legend()
plt.savefig('lstm_training_loss.png', dpi=300)
plt.close()

print("LSTM model completed:")
print("- Model saved as: lstm_model.h5")
print("- Evaluation results saved as: lstm_model_evaluation.csv")
print("- Prediction plot saved as: lstm_prediction_results.png")
print("- Training loss plot saved as: lstm_training_loss.png")
