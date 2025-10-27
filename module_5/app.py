import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# === Q4
# with open('pipeline_v1.bin', 'rb') as f:
#     model = pickle.load(f)

# === Q6
with open('pipeline_v2.bin', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# healthcheck
@app.get("/")
def read_root():
    return {"APP is running"}

@app.post("/predict")
@app.post("/predict")
def predict(client: Client):
    client_data = client.model_dump()
    
    probability = model.predict_proba([client_data])[0]
    
    return probability[1]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)