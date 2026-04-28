import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

base_path = "../../nasa-battery-dataset/cleaned_dataset/data"
file_path = "../../cleaned_dataset/data/00001.csv"  
metadata = pd.read_csv("../../cleaned_dataset/metadata.csv")

df = pd.read_csv(file_path)
discharge_meta = metadata[
    metadata['type'].str.lower() == 'discharge'
]
discharge_files = discharge_meta['filename'].astype(str).tolist()

capacities = []

def coulomb_capacity(time_s, current_a):
    time_h = time_s / 3600
    return np.trapz(np.abs(current_a), time_h)

capacity_i = coulomb_capacity(
    df['Time'].values,
    df['Current_measured'].values
)

for fname in discharge_files:
    file_path = f"{base_path}/{fname}"
    df = pd.read_csv(file_path)

    cap = coulomb_capacity(
        df['Time'].values,
        df['Current_measured'].values
    )
    capacities.append(cap)

capacities = np.array(capacities)
SOH_cc = capacities / capacities[0]

plt.figure(figsize=(8,4))
plt.plot(SOH_cc, marker='o')
plt.xlabel("Discharge Cycle Index")
plt.ylabel("SOH")
plt.title("SOH via Coulomb Counting")
plt.grid(True)
plt.show()


# VOLTAGE BASED SOH ESTIMATOR USING ML

def extract_voltage_features(df):
    features = {}
    
    v = df['Voltage_measured'].values
    t = df['Time'].values
    
    features['V_mean'] = v.mean()
    features['V_min'] = v.min()
    features['V_std'] = v.std()
    features['V_area'] = np.trapezoid(v, t)
    
    return features

X_voltage = []
y_soh = []

for i, fname in enumerate(discharge_files):
    df = pd.read_csv(f"{base_path}/{fname}")
    
    feats = extract_voltage_features(df)
    X_voltage.append(list(feats.values()))
    y_soh.append(SOH_cc[i])  # Coulomb-counted SOH

X_voltage = np.array(X_voltage)
y_soh = np.array(y_soh)


# CREATE TRAINING DATA

X_train, X_test, y_train, y_test = train_test_split(
    X_voltage, y_soh, test_size=0.2, random_state=42
)

# MODEL
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)          # ← this line was missing

y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")          # also worth printing this

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label='Perfect prediction') # diagonal line helps visualise accuracy
plt.xlabel("True SOH (Coulomb)")
plt.ylabel("Predicted SOH (Voltage-based)")
plt.title(f"Voltage-based SOH Estimation  |  MAE={mae:.4f}")
plt.legend()
plt.grid(True)
plt.show()
