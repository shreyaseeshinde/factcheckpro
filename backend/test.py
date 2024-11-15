from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer at startup
model_path = "./bert_classifier"

# Load model and set to evaluation mode
model = BertForSequenceClassification.from_pretrained(model_path)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# Define request schema
class TextInput(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    # Additional startup tasks can be defined here
    print("Model and tokenizer loaded successfully!")

@app.post("/predict")
async def predict(input_data: TextInput):
    # Tokenize input
    inputs = tokenizer(input_data.text, return_tensors="pt", padding="max_length", truncation=True)
    inputs = {key: val for key, val in inputs.items()}

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Return prediction as JSON
    return {"prediction": int(predictions[0])}
