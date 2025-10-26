# HTF25-Team-414

Lightweight social-post drafting app (frontend + AI-backed moderation & drafting backend).

This repository contains a Vite + React (TypeScript) frontend and a small Python FastAPI backend that runs local models and (optionally) calls Hugging Face Inference API. The app performs moderation and sentiment analysis on drafts before allowing a post to be published.

## Contents

- `src/` — frontend React app (Vite + TypeScript)
- `backend/` — FastAPI server, local model fallback and inference helpers
- `package.json`, `pnpm-lock.yaml` / `bun.lockb` — frontend package metadata

## Prerequisites

- Node.js 18+ (or your preferred LTS) and a package manager (npm, pnpm or yarn)
- Python 3.10+ and virtualenv / venv
- Optional: Hugging Face API token (only if you want to use a hosted HF model)

## Quick start (local development)

This guide assumes you run frontend on port 5173 (default Vite) and backend on port 8000.

1. Start the backend

```bash
# from repo root
cd backend
# create a venv (macOS / Linux)
python -m venv .venv
source .venv/bin/activate
# install dependencies
pip install -r requirements.txt

# Optionally set these environment variables (replace values):
# export HF_TOKEN="<your-hf-token>"
# export HF_REMOTE_MODEL="meta-llama/Llama-2-13b-chat-hf"  # example
# export PORT=8000

# Run the server (uvicorn)
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Watch the backend logs. On startup it prints which local model is used and whether a hosted fallback is configured.

2. Start the frontend

```bash
# from repo root
# install frontend deps (choose npm/pnpm/yarn)
npm install
# set the API base so frontend talks to the backend
echo "VITE_API_URL=http://localhost:8000" > .env

# run dev server
npm run dev
```

Open http://localhost:5173 (or the printed Vite URL) and try creating a post.

## How moderation & posting flow works

- The frontend calls the backend endpoint `/api/analyze-post` to get moderation and sentiment results for a candidate post. The UI shows these badges while typing.
- Create/Send is blocked until the analysis is performed. If the backend returns verdict `blocked` or `flagged`, the frontend prevents publishing and shows an error toast.
- The backend uses a small local fallback model by default (configured within `backend/main.py`). You can configure it to use a hosted model by setting `HF_TOKEN` and `HF_REMOTE_MODEL`.

Important endpoints (backend)

- `POST /api/analyze-post` — analyze a text and return moderation + sentiment (used before posting)
- `POST /api/moderate` — a simpler moderation-only endpoint
- `POST /api/draft-post` — generate a draft / rewrite using AI
- `POST /api/summarize` — summarization endpoint used for threads

Note: The frontend's `src/services/aiService.ts` expects the backend base URL from `VITE_API_URL`.

## Example curl

Analyze a sample text from the terminal:

```bash
curl -s -X POST http://localhost:8000/api/analyze-post \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a test post"}' | jq
```

Generate a draft:

```bash
curl -s -X POST http://localhost:8000/api/draft-post \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short announcement about a new product launch"}' | jq
```

## Configuration & environment variables

- `VITE_API_URL` — frontend: base URL for backend API (e.g., `http://localhost:8000`)
- `HF_TOKEN` — backend: (optional) Hugging Face API token to use hosted models
- `HF_REMOTE_MODEL` — backend: (optional) HF model id to call via inference API (e.g. `tiiuae/falcon-40b-instruct`)
- `PORT` — backend port (default 8000)

Add these to your shell or a `.env` file (for development only). If you enable `HF_TOKEN` + `HF_REMOTE_MODEL` the backend will attempt to call the HF Inference API and fall back to the local small model on failure.

## Troubleshooting

- CORS errors: the backend already allows `http://localhost:5173` and `http://localhost:3000`. If you run the frontend on a different origin, add it in `backend/main.py` CORS middleware.
- Model loading is slow on first run. Local generation may be CPU-bound. The server prints logs showing which model is in use.
- If moderation always returns `safe` or analysis is failing, check backend logs for model-loading errors and ensure `requirements.txt` deps are installed (transformers, torch, etc.).

## Tests and linting

This project does not include automated tests in the repo root. Run TypeScript type checks and linting using your preferred tools (e.g., `npm run build` or `pnpm build`) before publishing.

## Deployment notes

- For production, build the frontend (`npm run build`) and serve the static assets from a CDN or a simple static server. Point `VITE_API_URL` at your production backend.
- The backend can be containerized; ensure models and weights are available or configure HF hosted inference for a managed option.

## Contributing

Contributions are welcome. Please open issues or PRs. Keep changes small and include tests where appropriate.

## License

MIT
