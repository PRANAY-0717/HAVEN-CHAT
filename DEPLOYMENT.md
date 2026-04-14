# 🛡️ Haven - Deployment Guide

Haven: A place of safety or refuge from the rest of the toxic internet.

## 🚀 Frontend (Vercel)

1. **Connect your GitHub Repository** to Vercel.
2. **Root Directory**: Select `bloom-chat`.
3. **Build Settings**: Vercel will auto-detect Next.js.
4. **Environment Variables**: Add the following (refer to `.env.example`):
   - `NEXT_PUBLIC_SUPABASE_URL`: Your Supabase Project URL.
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Your Supabase Anon Key.
   - `NEXT_PUBLIC_ML_API_URL`: The URL of your deployed ML API (e.g., `https://haven-api.onrender.com`).

## 🤖 ML API (Render / Railway)

The ML service is a FastAPI app located in `bloom-chat/ml-service`.

### Option 1: Render (Recommended)
1. New **Web Service**.
2. **Root Directory**: `ml-service`.
3. **Build Command**: `pip install -r requirements.txt && python train.py`.
4. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. **Environment Variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API Key.

## 🗄️ Database (Supabase)

1. Create a new project on [Supabase](https://supabase.com).
2. Run the queries in `supabase/schema.sql` in the SQL Editor.
3. Enable **Realtime** for the `messages` table in the Supabase Dashboard.
