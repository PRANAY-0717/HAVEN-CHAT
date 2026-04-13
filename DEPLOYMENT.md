# 🌸 Bloom Chat - Deployment Guide

This project is structured as a monorepo for easy deployment.

## 🚀 Frontend (Vercel)

1. **Connect your GitHub Repository** to Vercel.
2. **Root Directory**: Select `bloom-chat`.
3. **Build Settings**: Vercel will auto-detect Next.js.
4. **Environment Variables**: Add the following:
   - `NEXT_PUBLIC_SUPABASE_URL`: Your Supabase Project URL.
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Your Supabase Anon Key.
   - `NEXT_PUBLIC_ML_API_URL`: The URL of your deployed ML API (e.g., `https://your-ml-api.railway.app`).

## 🤖 ML API (Render / Railway / Google App Engine)

The ML service is a FastAPI app located in `bloom-chat/ml-service`.

### Option 1: Render (Recommended)
1. New **Web Service**.
2. **Root Directory**: `bloom-chat/ml-service`.
3. **Build Command**: `pip install -r requirements.txt && python train.py`.
4. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. **Environment Variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API Key.

### Option 2: Railway
1. New **Service** from GitHub.
2. Railway will detect the Python environment.
3. Add the `GEMINI_API_KEY` environment variable.
4. Update the **Root Directory** in settings to `bloom-chat/ml-service`.

---
**Note**: Ensure the `NEXT_PUBLIC_ML_API_URL` on Vercel matches your deployed API URL.
