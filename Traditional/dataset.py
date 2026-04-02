import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

metadata = pd.read_csv('../../nasa-battery-dataset/cleaned_dataset/metadata.csv')

discharge_meta = metadata[
    metadata['type'].str.lower() == 'discharge'
]

discharge_files = discharge_meta['filename'].astype(str).tolist()

discharge_meta = discharge_meta.copy()
discharge_meta['Capacity'] = pd.to_numeric(
    discharge_meta['Capacity'],
    errors='coerce'
)

discharge_meta = discharge_meta.dropna(subset=['Capacity'])
discharge_meta = discharge_meta.reset_index(drop=True)

capacity = discharge_meta['Capacity'].values

SOH = capacity / capacity[0]

plt.figure(figsize=(8,4))
plt.plot(SOH, linewidth=2)
plt.xlabel("Discharge Cycle Index")
plt.ylabel("SOH")
plt.title("Battery State of Health (SOH)")
plt.grid(True)
plt.show()

battery_id = discharge_meta['battery_id'].iloc[0]
battery_df = discharge_meta[
    discharge_meta['battery_id'] == battery_id
].copy()

battery_df = battery_df.sort_values('start_time').reset_index(drop=True)
capacity = battery_df['Capacity'].values
SOH = capacity / capacity[0]

plt.figure(figsize=(6,4))
plt.plot(SOH, marker='o')
plt.xlabel("Discharge Cycle")
plt.ylabel("SOH")
plt.title(f"SOH Degradation - Battery {battery_id}")
plt.grid(True)
plt.show()