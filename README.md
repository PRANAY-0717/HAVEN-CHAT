# 🌸 Bloom Chat

Bloom Chat is a production-grade, real-time chatroom with an integrated machine learning toxicity detection system. It features a polished FAANG-level UI/UX, real-time analytics, and AI-driven behavior insights.

## 🚀 Features

- **Real-time Chat**: Instant message delivery using Supabase Realtime.
- **AI Moderation**: Every message is analyzed for toxicity before being displayed.
- **Toxicity Meter**: Visual feedback while typing, showing the probability of your message being flagged.
- **Analytics Dashboard**: Community-wide statistics on safety and interaction.
- **AI Insights**: Personalized behavior analysis based on your communication style.
- **Glassmorphism UI**: Modern, sleek design with smooth Framer Motion animations.
- **Dark/Light Mode**: Fully responsive theme support.

## 🛠️ Tech Stack

- **Frontend**: Next.js 15 (App Router), TypeScript, Tailwind CSS, Framer Motion, shadcn/ui.
- **Backend**: Supabase (Auth, Database, Realtime).
- **ML Layer**: Python (FastAPI), scikit-learn (TF-IDF + Logistic Regression).

## 📦 Project Structure

```text
bloom-chat/
├── ml-service/          # Python FastAPI ML Service
│   ├── models/          # Trained model and vectorizer
│   ├── train.py         # Training script
│   └── main.py          # FastAPI server
├── src/
│   ├── app/             # Next.js App Router pages
│   ├── components/      # UI and functional components
│   ├── hooks/           # Custom React hooks
│   └── lib/             # Supabase and utility functions
└── supabase/            # SQL schema and policies
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
- Update `.env.local` with your credentials.

## 🧪 Demo Mode
If `NEXT_PUBLIC_SUPABASE_URL` is not provided in `.env.local`, the application will run in **Demo Mode** with a mocked Supabase client, allowing you to explore the UI and ML features instantly.

---
Built for the college project fair with ❤️ and FAANG-level polish.
