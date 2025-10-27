import pickle


# === Q1
# uv --version
# uv 0.9.5

# === Q2
# sha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e

# === Q3
with open('pipeline_v1.bin', 'rb') as f:
    model = pickle.load(f)

test_data_3 = [
    {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
]

for i, data in enumerate(test_data_3, 1):
    pred = model.predict([data])[0]
    proba = model.predict_proba([data])[0]
print(proba[1])
# 0.5336072702798061

