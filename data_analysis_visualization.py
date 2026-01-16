import pandas as pd
import matplotlib.pyplot as plt

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# 1. Descriptive Statistics
stats = data[['temperature', 'humidity', 'temperature_denoised', 'humidity_denoised']].describe()
stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved as: descriptive_stats.csv")

# 2. Time-Series Plots (Raw vs Denoised Data)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Temperature Trend
ax1.plot(data['timestamp'], data['temperature'], label='Raw Temperature', color='lightblue', alpha=0.6)
ax1.plot(data['timestamp'], data['temperature_denoised'], label='Denoised Temperature', color='red', linewidth=2)
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (℃)')
ax1.set_title('Temperature Time-Series Trend')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# Humidity Trend
ax2.plot(data['timestamp'], data['humidity'], label='Raw Humidity', color='lightgreen', alpha=0.6)
ax2.plot(data['timestamp'], data['humidity_denoised'], label='Denoised Humidity', color='darkgreen', linewidth=2)
ax2.set_xlabel('Time')
ax2.set_ylabel('Humidity (%)')
ax2.set_title('Humidity Time-Series Trend (No Watering for 2 Days)')
ax2.legend()
ax2.tick_params(axis='x', rotation=45)

# Adjust layout and save
plt.tight_layout()
plt.savefig('time_series_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Time-series plots saved as: time_series_plots.png")
