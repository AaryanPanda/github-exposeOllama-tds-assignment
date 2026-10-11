import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- Config ----
# Groq is OpenAI-compatible. We forward the grader's request to it, adding our
# secret API key (kept in an environment variable, never in this file) and
# forcing a known-good model so it works no matter what model name is sent.
UPSTREAM = "https://api.groq.com/openai/v1/chat/completions"
MODEL    = "llama-3.1-8b-instant"          # current free Groq model
API_KEY  = os.environ.get("GROQ_API_KEY")  # set this in Render's Environment tab

app = FastAPI()

# CORS: let the grader's web page call us from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()

    # Force our model and non-streaming, keep the grader's messages as-is
    body["model"] = MODEL
    body["stream"] = False

    # Forward to Groq with our key attached
    async with httpx.AsyncClient(timeout=60) as client:
        upstream = await client.post(
            UPSTREAM,
            json=body,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

    # Hand Groq's OpenAI-format reply straight back to the grader
    return JSONResponse(upstream.json(), status_code=upstream.status_code)


# A tiny root endpoint, handy for waking the service before grading.
@app.get("/")
def root():
    return {"status": "ok", "model": MODEL, "key_loaded": bool(API_KEY)}
