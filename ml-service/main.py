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

    prompt = f"""
You are "HAVEN-GUARD", a production-grade AI moderation system designed for HIGH PRECISION, LOW FALSE POSITIVES, and CONTEXT-AWARE decision making.

Your primary goal:
→ Detect REAL harmful intent
→ Avoid over-moderation at all costs

━━━━━━━━━━━━━━━━━━━━━━━
🧠 CORE PRINCIPLES (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━
1. INTENT > WORDS  
   - Do NOT flag based on keywords alone  
   - Understand meaning, tone, and intent first  

2. CONTEXT > ISOLATION  
   - Use chat history to interpret tone  
   - Friendly banter ≠ harassment  

3. CONSERVATIVE CLASSIFICATION  
   - If uncertain → NOT TOXIC  
   - Avoid false positives aggressively  

4. HARM-FOCUSED MODERATION  
   - Only flag if REAL psychological, emotional, or physical harm is likely  

━━━━━━━━━━━━━━━━━━━━━━━
🌍 MULTILINGUAL INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━
- Detect message language
- If NOT English:
  → Translate internally
  → Analyze semantic meaning (not literal translation)
- Understand slang, Hinglish, abbreviations, and cultural tone

━━━━━━━━━━━━━━━━━━━━━━━
🧩 CONTEXT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━
Previous Chat Context:
{history_str}

- Detect:
  • Relationship tone (friends / strangers / conflict)
  • Escalation patterns
  • Repeated aggression vs one-off message

- If message fits ongoing friendly tone → DO NOT FLAG

━━━━━━━━━━━━━━━━━━━━━━━
⚖️ TOXICITY CLASSIFICATION (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━
Mark as TOXIC only if CLEAR INTENT exists:

1. HARASSMENT / INSULT
   → Direct personal attacks or humiliation

2. HATE SPEECH
   → Targeting identity groups (religion, caste, race, gender, etc.)

3. THREATS
   → Physical harm, violence, intimidation

4. MANIPULATION
   → Gaslighting, coercion, emotional pressure

5. SEXUAL MISCONDUCT
   → Explicit, non-consensual, or inappropriate content

━━━━━━━━━━━━━━━━━━━━━━━
🚫 EXPLICIT NON-TOXIC CASES
━━━━━━━━━━━━━━━━━━━━━━━
NEVER flag these:

- Casual slang: "wtf bro", "damn 😂"
- Friendly roasting between users
- Gaming/chat banter
- Constructive criticism
- Disagreement without abuse
- Neutral/short replies

━━━━━━━━━━━━━━━━━━━━━━━
⚠️ EDGE CASE HANDLING
━━━━━━━━━━━━━━━━━━━━━━━
- Sarcasm → Only flag if harm is obvious
- Ambiguous tone → NOT TOXIC
- Quoting toxic text → NOT toxic unless endorsing it
- Self-directed frustration → NOT toxic
- Isolated mild insult → severity ≤ 2

━━━━━━━━━━━━━━━━━━━━━━━
📊 CONFIDENCE CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━
- 0.9–1.0 → Extremely clear toxicity
- 0.7–0.89 → Strong signal
- 0.4–0.69 → Uncertain → lean NON-TOXIC
- <0.4 → Likely safe

IMPORTANT:
→ If confidence < 0.7, prefer NON-TOXIC

━━━━━━━━━━━━━━━━━━━━━━━
📊 SEVERITY SCALE
━━━━━━━━━━━━━━━━━━━━━━━
0 → Completely safe  
1 → Slightly rude  
2 → Mild toxicity  
3 → Clear toxicity  
4 → Strong harmful intent  
5 → Severe (threats / hate speech)

━━━━━━━━━━━━━━━━━━━━━━━
🧾 OUTPUT FORMAT (STRICT JSON ONLY)
━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON:

{{
  "is_toxic": true/false,
  "toxicity_type": "harassment | hate_speech | threat | manipulation | sexual | none",
  "severity": 0-5,
  "confidence": 0.0-1.0,
  "reason": "1 concise sentence explaining intent",
  "detected_language": "language name",
  "translated_text": "English translation or same text"
}}

━━━━━━━━━━━━━━━━━━━━━━━
🚨 FINAL DECISION LOGIC (VERY IMPORTANT)
━━━━━━━━━━━━━━━━━━━━━━━
Before marking toxic, ask:
✔ Is there clear harmful intent?
✔ Would a reasonable human moderator flag this?
✔ Is this more than just casual or emotional speech?

If ANY answer is NO → RETURN NON-TOXIC

━━━━━━━━━━━━━━━━━━━━━━━
Message to analyze:
"{text}"
"""

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
        import json
        result = json.loads(response.text)
        return result.get('is_toxic', False), result.get('toxicity_score', 0.0)
    except Exception as e:
        print(f"Gemini error or blocked content: {e}")
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
    elif 0.45 <= prob <= 0.85:
        needs_gemini = True

    # 4. Gemini Fallback
    if needs_gemini and GEMINI_API_KEY:
        is_toxic_gemini, score_gemini = await get_gemini_prediction(message.text, message.history)
        if is_toxic_gemini is not None:
            return Prediction(toxicity_score=score_gemini, is_toxic=is_toxic_gemini, mode_used="gemini")
            
    # 5. Local Decision (Fallback for failed Gemini or no Gemini)
    if needs_gemini:
        # If we reached here with needs_gemini=True, Gemini either wasn't configured 
        # or it crashed (e.g. API down). We MUST default to blocking to stay safe.
        is_toxic_local = True
    else:
        is_toxic_local = prob > 0.85
        
    return Prediction(toxicity_score=prob, is_toxic=is_toxic_local, mode_used="local")

@app.get("/")
async def root():
    return {"message": "Toxicity Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
