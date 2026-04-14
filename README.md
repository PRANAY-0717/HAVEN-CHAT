# 🛡️ Haven

**Haven**: A place of safety or refuge from the rest of the toxic internet.

Haven is a production-grade, real-time chatroom with an integrated machine learning toxicity detection system. It features a polished FAANG-level UI/UX, real-time analytics, and AI-driven behavior insights, all within a safe, moderated environment.

## 🚀 Features

- **Real-time Chat**: Instant message delivery using Supabase Realtime.
- **AI Moderation**: Dual-mode toxicity detection (Local Logistic Regression or Google Gemini 1.5 Flash).
- **Safe Haven Cleanup**: Chatroom is automatically cleaned every 20 minutes to keep the conversation fresh.
- **Message Collections**: Save important messages to your personal collection to protect them from the auto-cleanup.
- **Toxicity Meter**: Visual feedback while typing, showing the probability of your message being flagged.
- **Analytics Dashboard**: Community-wide statistics on safety and interaction.
- **AI Insights**: Personalized behavior analysis based on your communication style.
- **Animated Background**: Dynamic, floating glassmorphism elements with a technical grid overlay.
- **Dark/Light Mode**: Fully responsive theme support.

## 🛠️ Tech Stack

- **Frontend**: Next.js 15 (App Router), TypeScript, Tailwind CSS, Framer Motion, shadcn/ui.
- **Backend**: Supabase (Auth, Database, Realtime).
- **ML Layer**: Python (FastAPI), scikit-learn (TF-IDF + Logistic Regression), Google Gemini API.

## 📦 Project Structure

```text
haven-chat/
├── ml-service/          # Python FastAPI ML Service
│   ├── models/          # Trained model and vectorizer
│   ├── train.py         # Training script (improved with tri-grams)
│   └── main.py          # FastAPI server with robust error handling
├── src/
│   ├── app/             # Next.js App Router pages
│   ├── components/      # UI and functional components
│   ├── hooks/           # Custom React hooks
│   └── lib/             # Supabase and utility functions
└── supabase/            # SQL schema (idempotent) and policies
```

## 🚦 Getting Started

### 1. ML Service Setup
```bash
cd ml-service
python3 -m pip install -r requirements.txt
python3 train.py
python3 main.py
```

### 2. Frontend Setup
```bash
npm install
npm run dev
```

### 3. Supabase Setup
- Create a new project on [Supabase](https://supabase.com).
- Execute the SQL in `supabase/schema.sql` in the SQL Editor.
- Update `.env.local` with your credentials (refer to `.env.example`).

---
Built with ❤️ for a safer internet experience.
