import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Inference Server")

class Input(BaseModel):
    X: list

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict")
def predict(inp: Input):
    return {"predictions": [1 for _ in inp.X]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)