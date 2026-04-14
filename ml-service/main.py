from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from typing import Optional
import uvicorn
import os

app = FastAPI(title="Haven Toxicity Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        gemini_model = None
else:
    print("GEMINI_API_KEY not found. High accuracy mode will be unavailable.")
    gemini_model = None

# Load local model and vectorizer
MODEL_PATH = 'models/model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Run train.py first.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

class Message(BaseModel):
    text: str
    mode: Optional[str] = "local" # "local" or "high_accuracy"

class Prediction(BaseModel):
    toxicity_score: float
    is_toxic: bool
    mode_used: str

async def get_gemini_prediction(text: str):
    prompt = f"""You are a high-precision toxicity detection AI for a safe community chat called "Haven".
    Your task is to analyze the following message and determine if it is toxic.

    DEFINITION OF TOXICITY:
    - Harassment, insults, or intent to cause emotional harm.
    - Hate speech (targeting race, religion, gender, etc.).
    - Threats of violence or self-harm.
    - Subtle manipulation, gaslighting, or microaggressions.
    - Explicit sexual content.

    NON-TOXIC EXAMPLES (Do NOT flag these):
    - "okok", "ok", "yes", "hello", "lol", "wow" (Conversational fillers)
    - "this is sick!", "f**k yeah!" (Slang used for excitement/positivity)
    - "i disagree with you" (Respectful disagreement)
    - "my bad", "no worries" (Polite conversation)

    SUBTLE TOXIC EXAMPLES (Flag these):
    - "You are overreacting, it's not a big deal." (Gaslighting)
    - "You're so well-spoken for someone like you." (Microaggression)
    - "Maybe try using your brain for once?" (Insult)

    INSTRUCTIONS:
    - Be conservative with false positives.
    - If the intent is clearly positive or neutral slang, set is_toxic to false.
    - Return ONLY a JSON object.

    Message to analyze: "{text}"

    JSON Output:
    {{
      "is_toxic": boolean,
      "toxicity_score": float (0.0 to 1.0)
    }}
    """
    try:
        response = gemini_model.generate_content(prompt)
        import json
        # Extract JSON from response text
        result_text = response.text.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        result = json.loads(result_text)
        return result['is_toxic'], result['toxicity_score']
    except Exception as e:
        print(f"Gemini error: {e}")
        return None, None

@app.post("/predict", response_model=Prediction)
async def predict(message: Message):
    if not message.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # High Accuracy Mode (Gemini)
    if message.mode == "high_accuracy" and GEMINI_API_KEY:
        is_toxic, score = await get_gemini_prediction(message.text)
        if is_toxic is not None:
            return Prediction(toxicity_score=score, is_toxic=is_toxic, mode_used="gemini")
    
    # Fallback to Local Mode
    X_tfidf = vectorizer.transform([message.text])
    prob = model.predict_proba(X_tfidf)[0][1]
    is_toxic = bool(model.predict(X_tfidf)[0])
    
    return Prediction(toxicity_score=prob, is_toxic=is_toxic, mode_used="local")

@app.get("/")
async def root():
    return {"message": "Toxicity Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
