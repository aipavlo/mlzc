import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv('data-files/course_lead_scoring.csv')

numerical_features = df.select_dtypes(include=np.number).columns.tolist()
categorical_features = df.select_dtypes(include="object").columns.tolist()

df[numerical_features] = df[numerical_features].fillna(0)
df[categorical_features] = df[categorical_features].fillna("NA")

full_train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)
train_df, val_df = train_test_split(full_train_df, test_size=0.25, random_state=1)
len(train_df), len(val_df), len(test_df)

target = "converted"
features = numerical_features + categorical_features
features.remove(target)

X_train = train_df[features].reset_index(drop=True)
y_train = train_df[target].reset_index(drop=True)

X_val = val_df[features].reset_index(drop=True)
y_val = val_df[target].reset_index(drop=True)

X_test = test_df[features].reset_index(drop=True)
y_test = test_df[target].reset_index(drop=True)

# === Q1
numerical = numerical_features.copy()
numerical.remove(target)

print("q1:")

for feat in numerical:
    auc = roc_auc_score(y_train, X_train[feat])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -X_train[feat])
    print(f"{feat:<25}: AUC: {auc:.3f}")
# number_of_courses_viewed: AUC: 0.764

# === Q2
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(X_train.to_dict(orient="records"))
X_val = dv.transform(X_val.to_dict(orient="records"))

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)
print(f"q2: {auc:.3f}")
# 0.817

# === Q3
thresholds = np.linspace(start=0.0, stop=1.0, num=101)
scores = []

for t in thresholds:
    y_val_binary = (y_pred >= t).astype(int)
    p = precision_score(y_val, y_val_binary, zero_division=0)
    r = recall_score(y_val, y_val_binary)
    scores.append((t, p, r))

df_scores = pd.DataFrame(scores, columns=["threshold", "precision", "recall"])
df_scores["pr_diff"] = np.abs(df_scores["precision"] - df_scores["recall"])

intersection_threshold = (
    df_scores.query("precision != 0 & recall != 0")
    .sort_values(by="pr_diff")
    .head(1)["threshold"]
    .values[0]
)

print(f"q3: {intersection_threshold:.3f}")
# q3: 0.640

# === Q4
df_scores["f1_score"] = (
    2 * (df_scores.precision * df_scores.recall) / (df_scores.precision + df_scores.recall)
)

best_f1_threshold = df_scores.iloc[np.argmax(df_scores.f1_score)]["threshold"]

print(f"q4: {best_f1_threshold:.3f}")
# q4: 0.570

# === Q5
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

auc_scores = []

for train_idx, val_idx in kfold.split(full_train_df):
    df_train_fold = full_train_df.iloc[train_idx]
    df_val_fold = full_train_df.iloc[val_idx]

    y_train_fold = df_train_fold[target].to_numpy()
    y_val_fold = df_val_fold[target].to_numpy()

    dv_fold = DictVectorizer(sparse=False)

    X_train_fold = dv_fold.fit_transform(
        df_train_fold[features].to_dict(orient="records")
    )

    X_val_fold = dv_fold.transform(df_val_fold[features].to_dict(orient="records"))

    model_fold = LogisticRegression(solver="liblinear", C=1.0, max_iter=1_000)
    model_fold.fit(X_train_fold, y_train_fold)

    y_pred_fold = model_fold.predict_proba(X_val_fold)[:, 1]
    auc = roc_auc_score(y_val_fold, y_pred_fold)
    auc_scores.append(auc)

std_auc = np.std(auc_scores)
print(f"q5: {std_auc:.4f}")
# 0.0358

# === Q6
C_values = [0.000001, 0.001, 1.0]

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

cv_results = {}

print('q6:')
for C in C_values:
    auc_scores_fold = []

    for train_idx, val_idx in kfold.split(full_train_df):
        df_train_fold = full_train_df.iloc[train_idx]
        df_val_fold = full_train_df.iloc[val_idx]

        y_train_fold = df_train_fold[target].to_numpy()
        y_val_fold = df_val_fold[target].to_numpy()

        dv_fold = DictVectorizer(sparse=False)

        X_train_fold = dv_fold.fit_transform(
            df_train_fold[features].to_dict(orient="records")
        )

        X_val_fold = dv_fold.transform(df_val_fold[features].to_dict(orient="records"))

        model_fold = LogisticRegression(solver="liblinear", C=C, max_iter=1_000)
        model_fold.fit(X_train_fold, y_train_fold)

        y_pred_fold = model_fold.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_fold)
        auc_scores_fold.append(auc)

    mean_auc = np.mean(auc_scores_fold)
    std_auc = np.std(auc_scores_fold)

    cv_results[C] = {"mean_auc": mean_auc, "std_auc": std_auc}

    print(f"C={C:<8.6f}, AUC: {mean_auc:.3f}, Std: {std_auc:.3f}")
# 0.001