# backend/main.py
"""
AI-Powered Threads Backend
Rewrite + Condense endpoints improved to be character-limit aware.
"""

import os
import re
import json
import math
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Config
MAX_POST_CHARS = 1000
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HAS_CUDA = torch.cuda.is_available()
DTYPE = torch.bfloat16 if HAS_CUDA and torch.cuda.is_bf16_supported() else torch.float16
DEVICE = 0 if HAS_CUDA else -1
executor = ThreadPoolExecutor(max_workers=4)

# Models (initialized in load)
tokenizer = None
model = None
sentiment_pipe = None

def load():
    global tokenizer, model, sentiment_pipe
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto" if HAS_CUDA else None,
        dtype=DTYPE,
    )
    model.eval()

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=DEVICE
    )
    print("✅ Ready")

load()

# -------------------------
# Helpers
# -------------------------
def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or "")).strip()

def remove_emojis_and_symbols(s: str) -> str:
    """
    Remove common emoji / symbol Unicode ranges by checking character codepoints.
    """
    if not s:
        return s
    out_chars = []
    for ch in s:
        cp = ord(ch)
        # Emoji / symbol blocks (common ranges)
        if (
            0x1F300 <= cp <= 0x1F5FF
            or 0x1F600 <= cp <= 0x1F64F
            or 0x1F680 <= cp <= 0x1F6FF
            or 0x1F700 <= cp <= 0x1F77F
            or 0x2600  <= cp <= 0x26FF
            or 0x2700  <= cp <= 0x27BF
            or 0xFE00  <= cp <= 0xFE0F
            or 0x1F900 <= cp <= 0x1F9FF
            or 0x1FA70 <= cp <= 0x1FAFF
        ):
            continue
        out_chars.append(ch)
    return ''.join(out_chars)

def trim_to_sentence_boundary(text: str, limit: int) -> str:
    """Trim text to last sentence-ending punctuation before limit, otherwise fall back to last space."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    # find last sentence terminator
    last_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    if last_end != -1 and last_end >= int(limit * 0.5):
        result = truncated[:last_end+1].strip()
    else:
        # fall back to last space
        result = truncated.rsplit(' ', 1)[0].strip()
    # Ensure not empty
    if not result:
        result = truncated[:limit]
    # If it doesn't end with punctuation, add a period (safe)
    if not result.endswith(('.', '!', '?', '…')):
        result = result.rstrip(' -*') + '.'
    return result

# -------------------------
# Token-aware generate()
# -------------------------
def generate(prompt: str, max_new: int = 256, use_sampling: bool = False, char_limit: Optional[int] = None) -> str:
    """
    Generate text using the loaded model/tokenizer with token budgeting that respects a character limit.

    - char_limit: if provided, we compute a safe token budget so model output stays within that char limit.
    - We use the tokenizer to compute input token length and cap generation by tokenizer.model_max_length.
    """
    global model, tokenizer
    if not model or not tokenizer:
        return ""

    # Estimate tokens allowed for target output.
    # Prefer to estimate tokens based on characters: assume avg_chars_per_token ~ 4 (conservative).
    avg_chars_per_token = 4.0

    if char_limit is None:
        target_chars = MAX_POST_CHARS
    else:
        target_chars = char_limit

    target_new_tokens = max(8, int(math.ceil(target_chars / avg_chars_per_token)))

    # But user may also pass explicit max_new; respect the smaller of both.
    target_new_tokens = min(target_new_tokens, max_new)

    # Tokenize prompt to get input length and ensure we don't exceed model max
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32000)
    if HAS_CUDA:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    model_max = getattr(tokenizer, "model_max_length", None) or getattr(model.config, "n_positions", None) or 32000

    # Ensure input + generated tokens <= model_max (leave small margin)
    safe_max_new = max(8, min(target_new_tokens, max(8, model_max - input_len - 8)))
    gen_kwargs = {
        "max_new_tokens": int(safe_max_new),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if use_sampling:
        gen_kwargs.update(do_sample=True, temperature=0.85, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # decode only newly generated tokens
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return clean(generated)

# -------------------------
# Request models
# -------------------------
class AnalyzePostRequest(BaseModel):
    text: str
    media_flags: Optional[List[str]] = []

class SummarizeThreadRequest(BaseModel):
    texts: List[str] = []
    image_counts: Optional[List[int]] = []
    image_alts: Optional[List[str]] = []

# -------------------------
# Endpoints (root, analyze, moderate, summarize)
# -------------------------
@app.get("/")
def root():
    return {"status": "running", "model": MODEL_NAME, "device": "GPU" if HAS_CUDA else "CPU"}

@app.post("/api/analyze-post")
async def analyze_post(req: AnalyzePostRequest):
    text = clean(req.text)
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "confidence": 0.0},
            "moderation": {"verdict": "safe", "confidence": 0.0, "labels": [], "reason": "Empty", "raw_label": ""},
            "review_flag": False,
        }

    sentiment = {"label": "NEUTRAL", "confidence": 0.0}
    try:
        s = sentiment_pipe(text[:512])[0]
        sentiment = {"label": s["label"], "confidence": float(s["score"])}
    except:
        pass

    verdict = "safe"
    labels = []
    reason = "Safe"
    confidence = 0.0

    def mod_task():
        prompt = f"""You are a content moderator. Analyze if this post is:
- safe (normal content)
- flagged (needs review)
- blocked (clear violation)

Text: {text}

Respond ONLY with JSON: {{ "verdict": "safe"|"flagged"|"blocked", "labels": [], "reason": "", "confidence": 0.0-1.0 }}"""
        return generate(prompt, max_new=150, use_sampling=False)

    try:
        result = await asyncio.get_event_loop().run_in_executor(executor, mod_task)
        match = re.search(r'\{[^}]+\}', result)
        if match:
            data = json.loads(match.group())
            verdict = data.get("verdict", "safe").lower()
            labels = data.get("labels", [])
            reason = data.get("reason", "Safe")
            confidence = float(data.get("confidence", 0.0))
    except:
        pass

    if verdict not in ["safe", "flagged", "blocked"]:
        verdict = "safe"

    return {
        "sentiment": sentiment,
        "moderation": {
            "verdict": verdict,
            "confidence": max(0.0, min(1.0, confidence)),
            "labels": labels if isinstance(labels, list) else [],
            "reason": reason,
            "raw_label": labels[0] if labels else "",
        },
        "review_flag": verdict in ["flagged", "blocked"],
    }

@app.post("/api/moderate")
async def moderate(req: AnalyzePostRequest):
    result = await analyze_post(req)
    mod = result["moderation"]
    mod["review_flag"] = result["review_flag"]
    return mod

@app.post("/api/summarize-thread")
async def summarize_thread(req: SummarizeThreadRequest):
    texts = [clean(t) for t in req.texts if t]
    if not texts:
        return {"summary": ""}

    context = " ".join(texts)
    if req.image_counts:
        img_count = sum(max(0, c) for c in req.image_counts)
        if img_count > 0:
            context += f" [{img_count} image{'s' if img_count != 1 else ''}]"

    reply_count = len(texts) - 1
    char_count = len(context)

    if reply_count <= 2 and char_count < 500:
        target = "one sentence"
        max_chars = 100
    elif reply_count <= 5 or char_count < 1500:
        target = "2-3 sentences"
        max_chars = 180
    else:
        target = "3-4 sentences with different viewpoints"
        max_chars = 250

    def sum_task():
        prompt = f"""Summarize this in {target}. Use simple language.

Text: {context[:3000]}

Summary:"""
        return generate(prompt, max_new=150, use_sampling=True)

    try:
        summary = await asyncio.get_event_loop().run_in_executor(executor, sum_task)
        summary = re.sub(r'^(summary|the thread)\s*:?\s*', '', summary, flags=re.I).strip()

        if len(summary) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            result = []
            length = 0
            for sent in sentences:
                if length + len(sent) <= max_chars:
                    result.append(sent)
                    length += len(sent) + 1
                else:
                    break
            summary = " ".join(result) if result else sentences[0][:max_chars]

        return {"summary": summary or "Discussion"}
    except:
        return {"summary": context[:100]}

# -------------------------
# Condense endpoint (character-aware)
# -------------------------
@app.post("/api/condense-to-post")
async def condense(data: Dict[str, Any]):
    text = clean(data.get("text", ""))
    if not text:
        return {"draft": ""}

    word_count = len(text.split())
    target_ratio = 0.4 if word_count > 100 else 0.6
    target_words = max(int(word_count * target_ratio), 20)

    def task():
        # instruct model to stay within character limit
        prompt = f"""You are an expert writer who rewrites text to make it shorter, punchier, and more readable.

INSTRUCTIONS:
- Reduce length to roughly {target_words} words.
- Keep all essential meaning and logical flow.
- Ensure the ***final output fits entirely within {MAX_POST_CHARS} characters*** (including spaces and punctuation).
- Do NOT exceed that character limit.
- Remove fluff, repetition, or weak phrasing.
- Write naturally like a human (not bullet points unless original is a list).
- Output only the rewritten version, ending on a complete sentence.
- No emojis, no hashtags.

TEXT:
{text}

COMPRESSED VERSION (≤ {MAX_POST_CHARS} chars):"""
        # request token budget tuned to char limit
        return generate(prompt, max_new=512, use_sampling=True, char_limit=MAX_POST_CHARS)

    try:
        draft = await asyncio.get_event_loop().run_in_executor(executor, task)
        draft = remove_emojis_and_symbols(draft).strip()
        draft = re.sub(r'#\w+', '', draft)
        draft = clean(draft)
        # Final safeguard: trim gracefully only if model slightly exceeded limit
        if len(draft) > MAX_POST_CHARS:
            draft = trim_to_sentence_boundary(draft, MAX_POST_CHARS)
        return {"draft": draft}
    except Exception as e:
        print(f"Condense error: {e}")
        return {"draft": trim_to_sentence_boundary(text, MAX_POST_CHARS)}

# -------------------------
# Draft (rewrite) endpoint (character-aware)
# -------------------------
@app.post("/api/draft-post")
async def draft_post(data: Dict[str, Any]):
    prompt_text = clean(data.get("prompt", ""))
    if len(prompt_text) < 3:
        return {"draft": "", "error": "Prompt too short"}

    def task():
        prompt = f"""Rewrite the following text so it expresses the same meaning but uses different words and structure.

RULES:
- Preserve meaning and tone.
- Use significantly different phrasing.
- Sound fluent, natural, and human.
- Ensure the final output **fits entirely within {MAX_POST_CHARS} characters** (including spaces and punctuation).
- Do NOT exceed that character limit.
- No emojis, no hashtags.
- Output only the rewritten text and end on a complete sentence.

ORIGINAL:
{prompt_text}

REWRITTEN (≤ {MAX_POST_CHARS} chars):"""
        return generate(prompt, max_new=512, use_sampling=True, char_limit=MAX_POST_CHARS)

    try:
        draft = await asyncio.get_event_loop().run_in_executor(executor, task)
        draft = remove_emojis_and_symbols(draft)
        draft = re.sub(r'#\w+', '', draft)
        draft = clean(draft)
        if len(draft) > MAX_POST_CHARS:
            draft = trim_to_sentence_boundary(draft, MAX_POST_CHARS)
        return {"draft": draft}
    except Exception as e:
        print(f"Draft error: {e}")
        return {"draft": trim_to_sentence_boundary(prompt_text, MAX_POST_CHARS)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
