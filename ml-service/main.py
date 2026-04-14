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
    history: Optional[list[str]] = []

class Prediction(BaseModel):
    toxicity_score: float
    is_toxic: bool
    mode_used: str

async def get_gemini_prediction(text: str, history: list[str] = []):
    history_str = "\n".join([f"- {h}" for h in history]) if history else "No previous context."
    prompt = f"""You are a high-precision toxicity detection AI for a safe community chat called "Haven".
    Your task is to analyze the following message and determine if it is toxic.
    
    Previous Chat Context:
    {history_str}

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
    - Take the previous chat context into account. A word might be toxic isolated but safe in context.
    - Be conservative with false positives.
    - If the intent is clearly positive or neutral slang, set is_toxic to false.
    - Return ONLY a JSON object.

    Message to analyze: "{text}"
    """
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        import json
        result = json.loads(response.text)
        return result.get('is_toxic', False), result.get('toxicity_score', 0.0)
    except Exception as e:
        print(f"Gemini error: {e}")
        return None, None


@app.post("/predict", response_model=Prediction)
async def predict(message: Message):
    if not message.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # 1. Whitelist / Heuristic Fast-Pass
    whitelist = ['kill the process', 'crash the server', 'execute the program', 'dump the memory', 'headshot', 'gg', 'gg wp', 'destroy the base']
    if message.text.lower().strip() in whitelist:
        return Prediction(toxicity_score=0.0, is_toxic=False, mode_used="whitelist")

    # 2. Local Estimation
    X_tfidf = vectorizer.transform([message.text])
    prob = model.predict_proba(X_tfidf)[0][1]
    
    # 3. Decision Fusion Engine
    needs_gemini = False
    if message.mode == "high_accuracy":
        needs_gemini = True
    elif 0.6 <= prob <= 0.85:
        needs_gemini = True

    # 4. Gemini Fallback
    if needs_gemini and GEMINI_API_KEY:
        is_toxic_gemini, score_gemini = await get_gemini_prediction(message.text, message.history)
        if is_toxic_gemini is not None:
            return Prediction(toxicity_score=score_gemini, is_toxic=is_toxic_gemini, mode_used="gemini")
    
    # 5. Local Decision
    is_toxic_local = prob > 0.85
    return Prediction(toxicity_score=prob, is_toxic=is_toxic_local, mode_used="local")

@app.get("/")
async def root():
    return {"message": "Toxicity Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
