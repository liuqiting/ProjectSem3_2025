import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load simulated data
data = pd.read_csv('simulated_sensor_data.csv')

# 1. Denoising: Moving average filter (window size = 3)
data['humidity_denoised'] = data['humidity'].rolling(window=3, center=True).mean()
data['temperature_denoised'] = data['temperature'].rolling(window=3, center=True).mean()

# Fill missing values (caused by rolling window)
data = data.fillna(method='ffill').fillna(method='bfill')

scaler = MinMaxScaler(feature_range=(0, 1))
data['humidity_norm'] = scaler.fit_transform(data[['humidity_denoised']])
data['temperature_norm'] = MinMaxScaler(feature_range=(0,1)).fit_transform(data[['temperature_denoised']])

# Save preprocessed data and scaler
data.to_csv('preprocessed_data.csv', index=False)
joblib.dump(scaler, 'scaler.pkl')

print("Data preprocessing completed:")
print("- Preprocessed data saved as: preprocessed_data.csv")
print("- Scaler saved as: scaler.pkl")
