from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from fastapi.middleware.cors import CORSMiddleware
from crud import UserCRUD
from models import User
from db import SessionDep

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow all origins. You can specify a list of allowed origins here.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model and tokenizer at startup
model_path = "./bert_classifier"

# Check if CUDA is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and set to evaluation mode, and move it to the selected device (GPU/CPU)
model = BertForSequenceClassification.from_pretrained(model_path).to(device)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)


# Define request schema
class Input(BaseModel):
    text: str


@app.on_event("startup")
async def load_model():
    # Additional startup tasks can be defined here
    print("Model and tokenizer loaded successfully!")


@app.post("/register")
async def create_user(data: User, session: SessionDep):
    user = UserCRUD.retrieve(session, data.username)
    if user:
        return {"success": False, "detail": "Username already exists"}

    hashed = generate_password_hash(data.password)
    user = User(
        name=data.name,
        username=data.username,
        password=hashed,
    )
    user = UserCRUD.create(session, user)
    del user.password
    return {"success": True, "detail": "User created successfully", "user": user}


@app.post("/login")
async def login(data: User, session: SessionDep):
    user = UserCRUD.retrieve(session, data.username)
    if not user:
        return {"success": False, "detail": "Username is not registered"}

    hashed = check_password_hash(user.password, data.password)

    if not hashed:
        return {"success": False, "detail": "Invalid password"}
    del user.password
    return {"success": True, "detail": "Login successful", "user": user}


@app.post("/predict")
async def predict(input: Input):
    labels = ["False", "Mostly False", "Mostly True", "True", "Unverified/Mixed"]

    # Tokenize input and move inputs to the correct device
    inputs = tokenizer(
        input.text, return_tensors="pt", padding="max_length", truncation=True
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move to GPU/CPU

    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    label = labels[predictions[0]]

    # Return prediction as JSON
    return {
        "success": True,
        "result": int(predictions[0]),
        "label": label,
    }
