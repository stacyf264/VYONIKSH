import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/processed/skyguard_anomaly_dataset.csv")

print(f"Total observations: {len(df)}")


# -----------------------------
# 2. Use ONLY normal data
# -----------------------------
# Your anomaly column:
# 0 = normal
# 1 = anomaly
#
# Since your anomalies were injected into temperature,
# we don't want those corrupted values to train the
# expected-value model.

normal_df = df[df["anomaly"] == 0].copy()

print(f"Normal observations used for training/testing: {len(normal_df)}")


# ============================================================
# 3. TEMPERATURE CORRECTION MODEL
#    Pressure + RH -> Expected Temperature
# ============================================================

X_temperature = normal_df[["p (mbar)", "rh (%)"]]
y_temperature = normal_df["T (degC)"]

valid_temperature = (
    X_temperature.notna().all(axis=1)
    & y_temperature.notna()
)

X_temperature = X_temperature.loc[valid_temperature]
y_temperature = y_temperature.loc[valid_temperature]


X_train_T, X_test_T, y_train_T, y_test_T = train_test_split(
    X_temperature,
    y_temperature,
    test_size=0.20,
    random_state=42
)

temperature_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

temperature_model.fit(X_train_T, y_train_T)

temperature_prediction = temperature_model.predict(X_test_T)

temperature_mae = mean_absolute_error(
    y_test_T,
    temperature_prediction
)

temperature_rmse = np.sqrt(
    mean_squared_error(y_test_T, temperature_prediction)
)

temperature_r2 = r2_score(
    y_test_T,
    temperature_prediction
)


# ============================================================
# 4. PRESSURE CORRECTION MODEL
#    Temperature + RH -> Expected Pressure
# ============================================================

X_pressure = normal_df[["T (degC)", "rh (%)"]]
y_pressure = normal_df["p (mbar)"]

valid_pressure = (
    X_pressure.notna().all(axis=1)
    & y_pressure.notna()
)

X_pressure = X_pressure.loc[valid_pressure]
y_pressure = y_pressure.loc[valid_pressure]


X_train_P, X_test_P, y_train_P, y_test_P = train_test_split(
    X_pressure,
    y_pressure,
    test_size=0.20,
    random_state=42
)

pressure_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

pressure_model.fit(X_train_P, y_train_P)

pressure_prediction = pressure_model.predict(X_test_P)

pressure_mae = mean_absolute_error(
    y_test_P,
    pressure_prediction
)

pressure_rmse = np.sqrt(
    mean_squared_error(y_test_P, pressure_prediction)
)

pressure_r2 = r2_score(
    y_test_P,
    pressure_prediction
)


# ============================================================
# 5. HUMIDITY CORRECTION MODEL
#    Temperature + Pressure -> Expected RH
# ============================================================

X_humidity = normal_df[["T (degC)", "p (mbar)"]]
y_humidity = normal_df["rh (%)"]

valid_humidity = (
    X_humidity.notna().all(axis=1)
    & y_humidity.notna()
)

X_humidity = X_humidity.loc[valid_humidity]
y_humidity = y_humidity.loc[valid_humidity]


X_train_H, X_test_H, y_train_H, y_test_H = train_test_split(
    X_humidity,
    y_humidity,
    test_size=0.20,
    random_state=42
)

humidity_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

humidity_model.fit(X_train_H, y_train_H)

humidity_prediction = humidity_model.predict(X_test_H)

humidity_mae = mean_absolute_error(
    y_test_H,
    humidity_prediction
)

humidity_rmse = np.sqrt(
    mean_squared_error(y_test_H, humidity_prediction)
)

humidity_r2 = r2_score(
    y_test_H,
    humidity_prediction
)


# ============================================================
# 6. DISPLAY MODEL PERFORMANCE
# ============================================================

print("\n========================================")
print("   VYONIKSH CORRECTION MODEL EVALUATION")
print("========================================")

print("\nTemperature Model")
print("----------------------------")
print(f"MAE  : {temperature_mae:.4f} °C")
print(f"RMSE : {temperature_rmse:.4f} °C")
print(f"R²   : {temperature_r2:.4f}")

print("\nPressure Model")
print("----------------------------")
print(f"MAE  : {pressure_mae:.4f} hPa")
print(f"RMSE : {pressure_rmse:.4f} hPa")
print(f"R²   : {pressure_r2:.4f}")

print("\nHumidity Model")
print("----------------------------")
print(f"MAE  : {humidity_mae:.4f} %")
print(f"RMSE : {humidity_rmse:.4f} %")
print(f"R²   : {humidity_r2:.4f}")


# ============================================================
# 7. GET NEW AWS OBSERVATION
# ============================================================

pressure = float(input("\nEnter Pressure (hPa): "))
humidity = float(input("Enter Relative Humidity (%): "))
temperature = float(input("Enter Temperature (degC): "))


# ============================================================
# 8. TEMPORARY ANOMALY FEATURE INPUT
# ============================================================
# This will later come automatically from SSCP.
# ============================================================

anomaly_sensor = input(
    "Which sensor has anomaly? "
    "(temperature/pressure/humidity): "
).strip().lower()


# ============================================================
# 9. Keep original AWS values
# ============================================================

corrected_temperature = temperature
corrected_pressure = pressure
corrected_humidity = humidity


# ============================================================
# 10. CORRECT ONLY THE ANOMALOUS SENSOR
# ============================================================

if anomaly_sensor == "temperature":

    temperature_input = pd.DataFrame([{
        "p (mbar)": pressure,
        "rh (%)": humidity
    }])

    expected_temperature = temperature_model.predict(
        temperature_input
    )[0]

    corrected_temperature = expected_temperature

    print("\nTemperature anomaly detected.")
    print(f"Expected temperature: {expected_temperature:.2f} °C")


elif anomaly_sensor == "pressure":

    pressure_input = pd.DataFrame([{
        "T (degC)": temperature,
        "rh (%)": humidity
    }])

    expected_pressure = pressure_model.predict(
        pressure_input
    )[0]

    corrected_pressure = expected_pressure

    print("\nPressure anomaly detected.")
    print(f"Expected pressure: {expected_pressure:.2f} hPa")


elif anomaly_sensor == "humidity":

    humidity_input = pd.DataFrame([{
        "T (degC)": temperature,
        "p (mbar)": pressure
    }])

    expected_humidity = humidity_model.predict(
        humidity_input
    )[0]

    corrected_humidity = expected_humidity

    print("\nHumidity anomaly detected.")
    print(f"Expected humidity: {expected_humidity:.2f} %")


else:

    print("\nNo valid anomaly sensor selected.")
    print("Values remain unchanged.")


# ============================================================
# 11. FINAL VALUES
# ============================================================

print("\n========================================")
print("        VYONIKSH CORRECTION")
print("========================================")

print("\nObserved AWS Values:")
print(f"T (degC) : {temperature:.2f} °C")
print(f"p (mbar) : {pressure:.2f} hPa")
print(f"rh (%)   : {humidity:.2f} %")

print("\nCorrected Values:")
print(f"T (degC) : {corrected_temperature:.2f} °C")
print(f"p (mbar) : {corrected_pressure:.2f} hPa")
print(f"rh (%)   : {corrected_humidity:.2f} %")

print("========================================")
