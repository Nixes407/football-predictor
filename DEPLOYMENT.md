# Deployment Guide

## Overview
- Frontend: Vercel (Next.js)
- Backend: Railway or Render (FastAPI)
- Database: PostgreSQL (Railway or Supabase)

## Step 1 — Database
1. Create a PostgreSQL database on Railway or Supabase
2. Copy the connection string
3. Set DATABASE_URL in your backend environment

## Step 2 — Backend (Railway)
1. Create a new Railway project
2. Connect your GitHub repo
3. Set root directory to: backend/
4. Set start command to:
   uvicorn main:app --host 0.0.0.0 --port $PORT
5. Add environment variables:
   DATABASE_URL=your_postgres_connection_string
   FRONTEND_URL=https://your-frontend.vercel.app

## Step 3 — Run Data Pipeline (once)
After backend is deployed, run these in order:

  python data/collector.py
  python features/builder.py
  python models/trainer.py

These populate the database and create the model
artifacts inside the backend deployment.

## Step 4 — Frontend (Vercel)
1. Import your GitHub repo into Vercel
2. Set root directory to: frontend/
3. Add environment variable:
   NEXT_PUBLIC_API_URL=https://your-backend.railway.app
4. Deploy

## Step 5 — Update vercel.json
Replace the placeholder backend URL in vercel.json
with your actual Railway backend URL.

## Local Development
Backend:

  cd backend
  pip install -r requirements.txt
  uvicorn main:app --reload

Frontend:

  cd frontend
  npm install
  npm run dev

Create backend/.env from backend/.env.example
Create frontend/.env.local:
  NEXT_PUBLIC_API_URL=http://localhost:8000
