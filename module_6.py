import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

df = pd.read_csv('data-files/car_fuel_efficiency.csv')

print(df.head())
print(df.isna().sum())

df = df.fillna(0)

y = df['fuel_efficiency_mpg']
X = df.drop('fuel_efficiency_mpg', axis=1)



df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['fuel_efficiency_mpg'].values
y_val = df_val['fuel_efficiency_mpg'].values
y_test = df_test['fuel_efficiency_mpg'].values

X_train = df_train.drop(columns=['fuel_efficiency_mpg'])
X_val = df_val.drop(columns=['fuel_efficiency_mpg'])
X_test = df_test.drop(columns=['fuel_efficiency_mpg'])

train_dicts = X_train.to_dict(orient='records')
val_dicts = X_val.to_dict(orient='records')
test_dicts = X_test.to_dict(orient='records')



dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")

dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)

importances = dt.feature_importances_
feature_names = dv.get_feature_names_out()

# === Q1
sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("q1:")
print(sorted_features[:5])
# vehicle_weight

# === Q2
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("q2:")
print(rmse)
# 0.4595777223092726


# === Q3
scores = []

for n in tqdm(range(10, 201, 10), desc="Training models"):
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    scores.append((n, rmse))
print("q3:")
for n, rmse in scores:
    print(f"n_estimators={n:<3}  RMSE={rmse:.3f}")
# 80

# === Q4
max_depth_values = [10, 15, 20, 25]

results = {}

for depth in tqdm(max_depth_values, desc="Testing max_depth"):
    rmses = []
    for n in range(10, 201, 10):
        rf = RandomForestRegressor(
            n_estimators=n,
            max_depth=depth,
            random_state=1,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)
    results[depth] = np.mean(rmses)
    print(f"max_depth={depth}, mean RMSE={np.mean(rmses):.3f}")
print("q4:")
print(results)
# 10

# === Q5
rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=20,
    random_state=1,
    n_jobs=-1
)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_names = dv.get_feature_names_out()

sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("q5:")
for name, score in sorted_features[:10]:
    print(f"{name}: {score:.4f}")
# vehicle_weight

# === Q6
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

watchlist = [(dtrain, 'train'), (dval, 'val')]

def train_and_eval(eta):
    params = {
        'eta': eta,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0
    }

    model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)
    y_pred = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

rmse_03 = train_and_eval(0.3)
rmse_01 = train_and_eval(0.1)

print("q6:")
print(f"RMSE (eta=0.3): {rmse_03:.3f}")
print(f"RMSE (eta=0.1): {rmse_01:.3f}")
# 0.1