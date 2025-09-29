import pandas as pd
import numpy as np

print(f"q1: {pd.__version__}")

df = pd.read_csv('data-files/car_fuel_efficiency.csv')
print(f"q2: {len(df.index)}")

count_distinct_fuel_type = df["fuel_type"].nunique()
print(f"q3: {count_distinct_fuel_type}")

n_cols_with_na = int(df.isna().any().sum())
print(f"q4: {n_cols_with_na}")

max_mpg_asia = df.loc[df["origin"] == "Asia", "fuel_efficiency_mpg"].max()
print(f"q5: {max_mpg_asia}")

hp = pd.to_numeric(df["horsepower"], errors="coerce")
median_before = hp.median()
m = hp.mode(dropna=True)
mode_value = m.iloc[0] if not m.empty else None
df["horsepower"] = hp if mode_value is None else hp.fillna(mode_value)
median_after = df["horsepower"].median()
print(f"q6: median_after - median_before = {median_after - median_before}")

asia = df.loc[df["origin"].eq("Asia")]
asia = asia[["vehicle_weight", "model_year"]]
asia = asia.head(7).copy()
X = asia.to_numpy(dtype=float)
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200], dtype=float)
w = XTX_inv @ X.T @ y
sum_w = float(w.sum())
print(f"q7: {sum_w}")