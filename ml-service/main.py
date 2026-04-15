from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
import json
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from typing import Optional
import uvicorn

app = FastAPI(title="Haven Toxicity Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Setup — use gemini-2.0-flash (1.5-flash is deprecated/404)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Gemini model (gemini-2.0-flash) configured successfully.")
    except Exception as e:
        print(f"⚠️ Error configuring Gemini: {e}")
        gemini_model = None
else:
    print("⚠️ GEMINI_API_KEY not found. Gemini fallback will be unavailable.")

# Load local model and vectorizer
MODEL_PATH = 'models/model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Run train.py first.")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

print(f"✅ Local model loaded successfully.")

class Message(BaseModel):
    text: str
    mode: Optional[str] = "local" # "local" or "high_accuracy"
    history: Optional[list[str]] = []

class Prediction(BaseModel):
    toxicity_score: float
    is_toxic: bool
    mode_used: str

# =============================================================================
# SAFE-PASS: Ultra-common greetings/phrases that should NEVER be flagged
# =============================================================================
SAFE_PHRASES = {
    'hi', 'hey', 'hello', 'howdy', 'yo', 'sup', 'hola',
    'how are you', 'how are you doing', 'how are you today',
    'how r u', 'how is it going', "how's it going",
    'how have you been', "how's your day", 'how is your day',
    'what are you up to', 'what are you doing',
    'good morning', 'good afternoon', 'good evening', 'good night',
    'gm', 'gn', 'bye', 'goodbye', 'see you', 'see ya',
    'thanks', 'thank you', 'thx', 'ty', 'ok', 'okay',
    'yes', 'no', 'yeah', 'nah', 'yep', 'nope', 'sure',
    'lol', 'lmao', 'haha', 'hehe', 'bruh', 'bro', 'dude',
    'nice', 'cool', 'wow', 'omg', 'gg', 'gg wp',
    'take care', 'have a nice day', 'have a great day',
    'good luck', 'best of luck', 'congrats', 'congratulations',
    'happy birthday', 'welcome', 'np', 'no problem',
    'hello bro', 'hello there', 'hey there', 'hi there',
    'what up', 'whats up', "what's up", 'wassup',
    'how do you do', 'long time no see', 'nice to meet you',
    'i agree', 'exactly', 'correct', 'true', 'right',
    'got it', 'makes sense', 'i see', 'i understand',
    'good job', 'well done', 'great work', 'awesome', 'brilliant',
    'hello bhai', 'hi bro', 'hey man', 'hey dude',
    'kaise ho', 'kya haal hai', 'theek hai', 'accha',
    'sahi hai yaar', 'kya baat hai', 'bohot accha',
    # Gaming/tech
    'kill the process', 'crash the server', 'execute the program',
    'dump the memory', 'headshot', 'destroy the base',
}

async def get_gemini_prediction(text: str, history: list[str] = []):
    """Use Gemini as a high-accuracy fallback for uncertain messages."""
    if not gemini_model:
        return None, None
        
    history_str = "\n".join([f"- {h}" for h in history]) if history else "No previous context."

    prompt = f"""You are "HAVEN-GUARD", a production-grade AI moderation system.

CRITICAL RULE: Your #1 priority is AVOIDING FALSE POSITIVES. 
Normal conversation must NEVER be flagged as toxic.

CORE PRINCIPLES:
1. INTENT > WORDS — Do NOT flag based on keywords alone
2. CONTEXT > ISOLATION — Use chat history to interpret tone  
3. IF UNCERTAIN → NOT TOXIC — Always err on the side of allowing messages
4. Only flag messages with CLEAR harmful intent toward a person

NEVER FLAG THESE (examples):
- Greetings: "hi", "hello", "how are you", "hey there"
- Questions: "where are you?", "what are you doing?"
- Positive: "you are great", "thank you", "good job"
- Neutral: "ok", "sure", "maybe", "I see"
- Frustration at things (not people): "I hate this bug", "this is annoying"
- Gaming/tech: "kill the process", "headshot", "gg"
- Slang: "bruh", "lol", "no cap", "that's fire"

ONLY FLAG IF ALL THREE ARE TRUE:
1. There is clear hostile intent directed at a person
2. A reasonable human moderator would flag this
3. It goes beyond casual/emotional speech into actual harm

Chat History:
{history_str}

Message to analyze: "{text}"

Return ONLY valid JSON:
{{
  "is_toxic": true/false,
  "toxicity_score": 0.0-1.0,
  "severity": 0-5,
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}"""

    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        result = json.loads(response.text)
        
        is_toxic = result.get('is_toxic', False)
        # Read toxicity_score first, fall back to confidence, then severity-derived
        score = result.get('toxicity_score', 
                result.get('confidence', 
                result.get('severity', 0) / 5.0))
        
        print(f"  Gemini verdict for \"{text}\": is_toxic={is_toxic}, score={score}")
        return is_toxic, score
    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        return None, None

@app.post("/predict", response_model=Prediction)
async def predict(message: Message):
    if not message.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    clean_text = message.text.lower().strip()
    
    # ─────────────────────────────────────────────────
    # LAYER 1: Safe-phrase fast-pass (instant, no ML)
    # ─────────────────────────────────────────────────
    if clean_text in SAFE_PHRASES:
        return Prediction(toxicity_score=0.0, is_toxic=False, mode_used="safe_pass")

    # Also pass very short messages (1-2 words, no profanity signals)
    words = clean_text.split()
    if len(words) <= 2 and not any(c in clean_text for c in ['!', '@', '#', '$', '*']):
        # Check if it's just a normal short phrase
        X_quick = vectorizer.transform([message.text])
        quick_prob = model.predict_proba(X_quick)[0][1]
        if quick_prob < 0.6:
            return Prediction(toxicity_score=quick_prob, is_toxic=False, mode_used="short_safe")

    # ─────────────────────────────────────────────────
    # LAYER 2: Local ML model estimation
    # ─────────────────────────────────────────────────
    X_tfidf = vectorizer.transform([message.text])
    prob = model.predict_proba(X_tfidf)[0][1]
    
    print(f"  Local model prob for \"{message.text}\": {prob:.4f}")
    
    # ─────────────────────────────────────────────────
    # LAYER 3: Decision routing
    # ─────────────────────────────────────────────────
    # LOW confidence toxic (<0.40) → definitely safe, skip Gemini
    if prob < 0.40:
        return Prediction(toxicity_score=prob, is_toxic=False, mode_used="local")
    
    # HIGH confidence toxic (>0.85) → definitely toxic, skip Gemini
    if prob > 0.85 and message.mode != "high_accuracy":
        return Prediction(toxicity_score=prob, is_toxic=True, mode_used="local")
    
    # UNCERTAIN zone (0.40 - 0.85) or high_accuracy mode → ask Gemini
    needs_gemini = True
    
    # ─────────────────────────────────────────────────
    # LAYER 4: Gemini fallback for uncertain cases
    # ─────────────────────────────────────────────────
    if needs_gemini and gemini_model and GEMINI_API_KEY:
        is_toxic_gemini, score_gemini = await get_gemini_prediction(
            message.text, message.history
        )
        if is_toxic_gemini is not None:
            return Prediction(
                toxicity_score=score_gemini, 
                is_toxic=is_toxic_gemini, 
                mode_used="gemini"
            )
    
    # ─────────────────────────────────────────────────
    # LAYER 5: Fallback — Gemini unavailable/failed
    # ─────────────────────────────────────────────────
    # CRITICAL FIX: When uncertain and Gemini fails,
    # default to NOT TOXIC (not blocking).
    # Only block if local model is very confident (>0.75)
    is_toxic_fallback = prob > 0.75
    
    print(f"  Gemini unavailable, fallback decision: prob={prob:.4f}, is_toxic={is_toxic_fallback}")
    
    return Prediction(
        toxicity_score=prob, 
        is_toxic=is_toxic_fallback, 
        mode_used="local_fallback"
    )

@app.get("/")
async def root():
    return {"message": "Haven Toxicity Detection API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
