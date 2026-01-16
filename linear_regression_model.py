import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Load data and scaler
data = pd.read_csv('preprocessed_data.csv')
scaler = joblib.load('scaler.pkl')

# Create sequences: Use 24 time steps (12 hours) to predict the next humidity value
def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data['humidity_norm'].iloc[i:i+seq_len].values)
        y.append(data['humidity_norm'].iloc[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(data, seq_len=24)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Inverse normalization to get real humidity values
y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Model evaluation
mse = mean_squared_error(y_test_denorm, y_pred_denorm)
mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
print(f"Linear Regression Model Evaluation:")
print(f"- Mean Squared Error (MSE): {mse:.2f}")
print(f"- Mean Absolute Error (MAE): {mae:.2f}")

# Save model and evaluation results
joblib.dump(lr_model, 'linear_regression_model.pkl')
eval_results = pd.DataFrame({
    'True_Humidity': y_test_denorm,
    'Predicted_Humidity': y_pred_denorm,
    'Error': y_test_denorm - y_pred_denorm
})
eval_results.to_csv('lr_model_evaluation.csv', index=False)

# Plot prediction results
plt.figure(figsize=(10, 4))
plt.plot(y_test_denorm, label='True Humidity', color='blue')
plt.plot(y_pred_denorm, label='LR Predicted Humidity', color='orange', linestyle='--')
plt.xlabel('Test Sample Index')
plt.ylabel('Humidity (%)')
plt.title('Linear Regression: 12-Hour Humidity Prediction Results')
plt.legend()
plt.savefig('lr_prediction_results.png', dpi=300)
plt.close()

print("Linear Regression model completed:")
print("- Model saved as: linear_regression_model.pkl")
print("- Evaluation results saved as: lr_model_evaluation.csv")
print("- Prediction plot saved as: lr_prediction_results.png")
