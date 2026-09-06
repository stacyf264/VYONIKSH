import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv("data/processed/skyguard_anomaly_dataset.csv")

# -----------------------------
# 2. Select the 3 AWS parameters
# -----------------------------
features = [
    "T (degC)",
    "p (mbar)",
    "rh (%)"
]

data = df[features].copy()

# Remove missing values
data = data.dropna()

# -----------------------------
# 3. Scale the three variables
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(data)

# -----------------------------
# 4. Train Isolation Forest
# -----------------------------
model = IsolationForest(
    contamination=0.01,
    random_state=42
)

model.fit(X)

# -----------------------------
# 5. Detect anomalies
# -----------------------------
data["model_prediction"] = model.predict(X)

# Isolation Forest:
#   1  = normal
#  -1  = anomaly

data["anomaly_status"] = np.where(
    data["model_prediction"] == -1,
    "Anomaly",
    "Normal"
)

# -----------------------------
# 6. Calculate anomaly score
# -----------------------------
data["anomaly_score"] = -model.score_samples(X)

# -----------------------------
# 7. Display results
# -----------------------------
print(data.head(20))

print("\nNumber of anomalies:")
print((data["model_prediction"] == -1).sum())

print("\nNumber of normal observations:")
print((data["model_prediction"] == 1).sum())

# -----------------------------
# 8. Save results
# -----------------------------
data.to_csv("anomaly_results.csv", index=False)

print("\nResults saved to anomaly_results.csv")
print("\nOriginal injected anomalies:")
print((df["anomaly"] == 1).sum())

print("\nModel detected anomalies:")
print((data["model_prediction"] == -1).sum())

print("\nInjected anomalies detected by model:")
print(
    (
        (df.loc[data.index, "anomaly"] == 1) &
        (data["model_prediction"] == -1)
    ).sum()
)