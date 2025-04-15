from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()
classifier = pipeline("text-classification", model="models/fine_tuned")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    return classifier(input.text)