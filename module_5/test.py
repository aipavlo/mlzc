import requests

url = "http://localhost:8000/predict"

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)

result = response.json()
print(result)

# === Q4
# 0.5340417283801275

# === Q5
# 181 MB (121?)

# === Q6
# 0.9933071490756734