import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps: 1 entry every 30 minutes for 2 days (96 entries total)
start_time = datetime(2025, 1, 1, 0, 0)
timestamps = [start_time + timedelta(minutes=30*i) for i in range(96)]

# Simulate temperature (15-28℃ with slight fluctuations)
temperature = np.random.normal(loc=22, scale=1.5, size=96)
temperature = np.clip(temperature, 15, 28)  # Restrict to reasonable range

# Simulate soil humidity (30%-70%, gradually decreasing without watering + noise)
humidity_trend = np.linspace(65, 35, 96)  # Decrease from 65% to 35% over 2 days
humidity_noise = np.random.normal(loc=0, scale=2, size=96)
humidity = humidity_trend + humidity_noise
humidity = np.clip(humidity, 30, 70)  # Restrict to reasonable range

# Create DataFrame and save as CSV
data = pd.DataFrame({
    'timestamp': [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
    'temperature': temperature.round(2),
    'humidity': humidity.round(2)
})
data.to_csv('simulated_sensor_data.csv', index=False)
print("Simulated sensor data generated successfully: simulated_sensor_data.csv")
