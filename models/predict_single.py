import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load existing dataset
# -----------------------------
df = pd.read_csv("data/processed/skyguard_anomaly_dataset.csv")

# -----------------------------
# 2. Select AWS parameters
# -----------------------------
features = [
    "T (degC)",
    "p (mbar)",
    "rh (%)"
]

X = df[features].dropna()

# -----------------------------
# 3. Scale the data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Train Isolation Forest
# -----------------------------
model = IsolationForest(
    contamination=0.01,
    random_state=42
)

model.fit(X_scaled)

# -----------------------------
# 5. Get observation from user
# -----------------------------
print("\n==============================")
print("       VYONIKSH")
print(" AWS Anomaly Detection System")
print("==============================\n")

temperature = float(input("Enter Temperature (°C): "))
pressure = float(input("Enter Pressure (hPa): "))
humidity = float(input("Enter Humidity (%): "))

# -----------------------------
# 6. Create new observation
# -----------------------------
new_data = pd.DataFrame([{
    "T (degC)": temperature,
    "p (mbar)": pressure,
    "rh (%)": humidity
}])

# -----------------------------
# 7. Scale using trained scaler
# -----------------------------
new_scaled = scaler.transform(new_data)

# -----------------------------
# 8. Predict
# -----------------------------
prediction = model.predict(new_scaled)[0]

# -----------------------------
# 9. Calculate anomaly score
# -----------------------------
score = -model.score_samples(new_scaled)[0]

# -----------------------------
# 10. Display result
# -----------------------------
if prediction == -1:
    status = "ANOMALY"
else:
    status = "NORMAL"

print("\n==============================")
print("       VYONIKSH RESULT")
print("==============================")

print(f"Temperature : {temperature} °C")
print(f"Pressure    : {pressure} hPa")
print(f"Humidity    : {humidity} %")

print(f"\nPrediction  : {status}")
print(f"Anomaly Score: {score:.4f}")

print("==============================\n")