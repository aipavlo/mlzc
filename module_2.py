import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, root_mean_squared_error

df = pd.read_csv('data-files/car_fuel_efficiency.csv')
features = ["engine_displacement", "horsepower", "vehicle_weight", "model_year"]
target   = "fuel_efficiency_mpg"
df = df[features + [target]]

# === Q1
na_counts = df[features].isna().sum()
print(f"q1: {na_counts}")

# === Q2
hp = pd.to_numeric(df["horsepower"], errors="coerce")
print(f"q2: {hp.median()}")

# === Q3
n = len(df)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
df_shuffled = df.iloc[idx].reset_index(drop=True)

n_train = int(n * 0.60)
n_val   = int(n * 0.20)
n_test  = n - n_train - n_val

df_train = df_shuffled.iloc[:n_train].reset_index(drop=True)
df_val   = df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
df_test  = df_shuffled.iloc[n_train + n_val:].reset_index(drop=True)

Xtr = df_train[features].apply(pd.to_numeric, errors="coerce")
Xva = df_val[features].apply(pd.to_numeric, errors="coerce")
ytr = pd.to_numeric(df_train[target], errors="coerce")
yva = pd.to_numeric(df_val[target],   errors="coerce")

mask_tr = ytr.notna()
mask_va = yva.notna()
Xtr, ytr = Xtr[mask_tr].reset_index(drop=True), ytr[mask_tr].reset_index(drop=True)
Xva, yva = Xva[mask_va].reset_index(drop=True), yva[mask_va].reset_index(drop=True)

# a)
Xa, Va = Xtr.copy(), Xva.copy()
Xa["horsepower"] = Xa["horsepower"].fillna(0.0)
Va["horsepower"] = Va["horsepower"].fillna(0.0)

m0 = LinearRegression().fit(Xa, ytr)
rmse0 = round(root_mean_squared_error(yva, m0.predict(Va)), 2)

# b)
mean_hp = Xtr["horsepower"].mean(skipna=True)
Xb, Vb = Xtr.copy(), Xva.copy()
Xb["horsepower"] = Xb["horsepower"].fillna(mean_hp)
Vb["horsepower"] = Vb["horsepower"].fillna(mean_hp)

mm = LinearRegression().fit(Xb, ytr)
rmsem = round(root_mean_squared_error(yva, mm.predict(Vb)), 2)

print("q3:", "With 0" if rmse0 < rmsem else ("With mean" if rmsem < rmse0 else "Both are equally good"))

# === Q4
Xtr_r = Xtr.fillna(0.0)
Xva_r = Xva.fillna(0.0)

r_list = [0, 0.01, 1, 10, 100]
scores_ridge = {}
for r in r_list:
    model = Ridge(alpha=r, fit_intercept=True).fit(Xtr_r, ytr)
    pred = model.predict(Xva_r)
    scores_ridge[r] = round(root_mean_squared_error(yva, pred), 2)

best_r = min(scores_ridge, key=lambda r: (scores_ridge[r], r))
print("q4:", best_r)

# === Q5
df_q5 = df[features + [target]].copy()
for c in features + [target]:
    df_q5[c] = pd.to_numeric(df_q5[c], errors="coerce")
df_q5 = df_q5[df_q5[target].notna()].reset_index(drop=True)

def prepare_X(dfs):
    return dfs[features].fillna(0.0).values

def rmse_for_seed(seed):
    n = len(df_q5)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    d = df_q5.iloc[idx].reset_index(drop=True)

    n_train = int(n * 0.60)
    n_val   = int(n * 0.20)

    dtr = d.iloc[:n_train].reset_index(drop=True)
    dva = d.iloc[n_train:n_train + n_val].reset_index(drop=True)

    X_train = prepare_X(dtr)
    y_train = dtr[target].values
    X_val   = prepare_X(dva)
    y_val   = dva[target].values

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return float(np.sqrt(mean_squared_error(y_val, y_pred)))

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = [rmse_for_seed(s) for s in seeds]

std_val = float(np.std(scores))
std_rounded = round(std_val, 3)
print("q5:", f"{std_rounded:.3f}")

# === Q6
df_q6 = df[features + [target]].copy()
for c in features + [target]:
    df_q6[c] = pd.to_numeric(df_q6[c], errors="coerce")
df_q6 = df_q6[df_q6[target].notna()].reset_index(drop=True)

n = len(df_q6)
idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)
d = df_q6.iloc[idx].reset_index(drop=True)

n_train = int(n * 0.60)
n_val   = int(n * 0.20)

dtr = d.iloc[:n_train].reset_index(drop=True)
dva = d.iloc[n_train:n_train + n_val].reset_index(drop=True)
dte = d.iloc[n_train + n_val:].reset_index(drop=True)

d_full = pd.concat([dtr, dva], ignore_index=True)

X_train_full = d_full[features].fillna(0.0).values
y_train_full = d_full[target].values
X_test = dte[features].fillna(0.0).values
y_test = dte[target].values

model = Ridge(alpha=0.001, fit_intercept=True)
model.fit(X_train_full, y_train_full)

y_pred_test = model.predict(X_test)
rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
rmse_test_rounded = round(rmse_test, 3)

print("q6:", rmse_test_rounded)