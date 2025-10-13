import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('data-files/course_lead_scoring.csv')
target = 'converted'

missing = df.isna().sum().sort_values(ascending=False)
print("Missing:\n", missing[missing > 0])

cat_cols = ['industry', 'lead_source', 'employment_status', 'location']
num_cols = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score']

df[cat_cols] = df[cat_cols].fillna('NA')
df[num_cols] = df[num_cols].fillna(0.0)


# === Q1
mode_industry = df['industry'].mode(dropna=False)[0]

print(f"q1:", {mode_industry})

# === Q2
corr = df[num_cols].corr(method='pearson')

print("q2:")
print(corr.to_string())
# annual_income & interaction_count = 0.027036

# === Q3
df_full_train, df_test = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df[target]
)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42, stratify=df_full_train[target]
)

y_train = df_train[target].values
y_val   = df_val[target].values
y_test  = df_test[target].values

df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

df_train = df_train.drop(columns=[target])
df_val   = df_val.drop(columns=[target])
df_test  = df_test.drop(columns=[target])

mi_scores = {}
for col in cat_cols:
    mi = mutual_info_score(y_train, df_train[col])
    mi_scores[col] = round(mi, 2)

print("q3:")
print(mi_scores)
# lead_source: 0.03

# === Q4
features = cat_cols + num_cols

dv = DictVectorizer(sparse=False)

train_dicts = df_train[features].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[features].to_dict(orient='records')
X_val = dv.transform(val_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"q4: {acc}")

# === Q5
def acc_without(feature_to_drop: str) -> float:
    kept = [f for f in features if f != feature_to_drop]
    dv_ = DictVectorizer(sparse=False)
    Xtr_ = dv_.fit_transform(df_train[kept].to_dict(orient='records'))
    Xvl_ = dv_.transform(df_val[kept].to_dict(orient='records'))
    mdl_ = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    mdl_.fit(Xtr_, y_train)
    yhat_ = mdl_.predict(Xvl_)
    return accuracy_score(y_val, yhat_)

candidates = ['industry', 'employment_status', 'lead_score']
diffs = {}

for f in candidates:
    acc_wo = acc_without(f)
    dif = acc - acc_wo  # baseline - accuracy_without
    diffs[f] = (acc_wo, dif)

print('q5')
print(diffs)
# employment_status

# === Q6
Cs = [0.01, 0.1, 1, 10, 100]

results = []
for c in Cs:
    model_c = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model_c.fit(X_train, y_train)
    acc = accuracy_score(y_val, model_c.predict(X_val))
    results.append((c, acc))

print('q6')
print(results)
# (0.01, 0.7337883959044369) - the best one '0.01'