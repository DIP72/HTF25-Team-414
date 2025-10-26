# backend/main.py - CONVERSATIONAL STYLE
"""
AI-Powered Threads Backend
Casual, readable summaries and rewrites
"""

import os
import re
import json
import math
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import uvicorn

# ================== CONFIG ==================
MAX_POST_CHARS = 1000
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_WORKERS = 4

# ================== GLOBAL RESOURCES ==================
HAS_CUDA = torch.cuda.is_available()
DTYPE = torch.bfloat16 if HAS_CUDA and torch.cuda.is_bf16_supported() else torch.float16
DEVICE = "cuda" if HAS_CUDA else "cpu"

tokenizer = None
model = None
sentiment_pipe = None
executor = None

# ================== GENERATION ==================
def generate_sync(prompt: str, max_new: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, char_limit: Optional[int] = None) -> str:
    """High-quality text generation"""
    global model, tokenizer
    if not model or not tokenizer:
        return ""

    try:
        avg_chars_per_token = 3.5
        if char_limit:
            target_new_tokens = int(math.ceil(char_limit / avg_chars_per_token * 1.3))
        else:
            target_new_tokens = max_new
        
        target_new_tokens = min(target_new_tokens, max_new)
        target_new_tokens = max(100, target_new_tokens)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        if HAS_CUDA:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": target_new_tokens,
            "min_new_tokens": 20,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return clean(generated)
    
    except Exception as e:
        print(f"Generation error: {e}")
        return ""

async def generate(prompt: str, max_new: int = 512, temperature: float = 0.7,
                   top_p: float = 0.9, char_limit: Optional[int] = None) -> str:
    """Async wrapper"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, generate_sync, prompt, max_new, temperature, top_p, char_limit
    )

# ================== MODEL LOADING ==================
def load_models():
    """Load models"""
    global tokenizer, model, sentiment_pipe, executor
    
    print(f"🚀 Loading {MODEL_NAME}...")
    
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto" if HAS_CUDA else None,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    
    if hasattr(torch, 'compile') and HAS_CUDA:
        print("⚡ Applying torch.compile...")
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            print("✅ torch.compile applied")
        except Exception as e:
            print(f"⚠️ torch.compile skipped: {e}")
    
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if HAS_CUDA else -1,
        batch_size=4,
        max_length=512,
        truncation=True
    )
    
    print(f"✅ Ready on {DEVICE.upper()}")

def cleanup_models():
    """Cleanup"""
    global executor
    if executor:
        executor.shutdown(wait=True)
    print("🛑 Shutdown complete")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    cleanup_models()

# ================== SETUP FASTAPI ==================
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== HELPERS ==================
def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or "")).strip()

def remove_emojis_and_symbols(s: str) -> str:
    if not s:
        return s
    out_chars = []
    for ch in s:
        cp = ord(ch)
        if (0x1F300 <= cp <= 0x1F5FF or 0x1F600 <= cp <= 0x1F64F or 
            0x1F680 <= cp <= 0x1F6FF or 0x1F700 <= cp <= 0x1F77F or 
            0x2600 <= cp <= 0x26FF or 0x2700 <= cp <= 0x27BF or 
            0xFE00 <= cp <= 0xFE0F or 0x1F900 <= cp <= 0x1F9FF or 
            0x1FA70 <= cp <= 0x1FAFF):
            continue
        out_chars.append(ch)
    return ''.join(out_chars)

def smart_truncate(text: str, limit: int, prefer_complete: bool = True) -> str:
    if len(text) <= limit:
        return text
    
    if prefer_complete and len(text) <= limit * 1.15:
        truncated = text[:limit]
        for end_char in ['. ', '! ', '? ']:
            last_pos = truncated.rfind(end_char)
            if last_pos >= limit * 0.85:
                return truncated[:last_pos + 1].strip()
        
        for end_char in [', ', '; ', ': ']:
            last_pos = truncated.rfind(end_char)
            if last_pos >= limit * 0.75:
                result = truncated[:last_pos].strip()
                if not result.endswith(('.', '!', '?')):
                    result += '.'
                return result
    
    truncated = text[:limit]
    sentence_ends = [truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?')]
    last_sentence = max(sentence_ends)
    
    if last_sentence != -1 and last_sentence >= int(limit * 0.7):
        return truncated[:last_sentence+1].strip()
    
    clause_ends = [truncated.rfind(','), truncated.rfind(';'), truncated.rfind(':')]
    last_clause = max(clause_ends)
    
    if last_clause != -1 and last_clause >= int(limit * 0.5):
        result = truncated[:last_clause].strip()
        if result and not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
    last_space = truncated.rfind(' ')
    if last_space != -1:
        result = truncated[:last_space].strip()
        if result and not result.endswith(('.', '!', '?', ',')):
            result += '.'
        return result
    
    return truncated.strip() + '.'

# ================== REQUEST MODELS ==================
class AnalyzePostRequest(BaseModel):
    text: str
    media_flags: Optional[List[str]] = []

class SummarizeThreadRequest(BaseModel):
    texts: List[str] = []
    image_counts: Optional[List[int]] = []
    image_alts: Optional[List[str]] = []

# ================== ENDPOINTS ==================
@app.get("/")
def root():
    return {"status": "running", "model": MODEL_NAME, "device": DEVICE.upper()}

@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE}

@app.post("/api/analyze-post")
async def analyze_post(req: AnalyzePostRequest):
    text = clean(req.text)
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "confidence": 0.0},
            "moderation": {"verdict": "safe", "confidence": 0.0, "labels": [], 
                          "reason": "Empty", "raw_label": ""},
            "review_flag": False,
        }

    sentiment_task = asyncio.create_task(_get_sentiment(text))
    moderation_task = asyncio.create_task(_get_moderation(text))
    
    sentiment, moderation = await asyncio.gather(sentiment_task, moderation_task)
    
    return {
        "sentiment": sentiment,
        "moderation": moderation,
        "review_flag": moderation["verdict"] in ["flagged", "blocked"],
    }

async def _get_sentiment(text: str) -> Dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: sentiment_pipe(text[:512], batch_size=1)[0]
        )
        return {"label": result["label"], "confidence": float(result["score"])}
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"label": "NEUTRAL", "confidence": 0.0}

async def _get_moderation(text: str) -> Dict[str, Any]:
    prompt = f"""Analyze this post for content moderation.

Categories:
- safe: Normal content
- flagged: Needs review
- blocked: Clear violation

Post: {text}

Respond with JSON:
{{"verdict": "safe"|"flagged"|"blocked", "labels": [], "reason": "", "confidence": 0.0-1.0}}

JSON:"""
    
    try:
        result = await generate(prompt, max_new=200, temperature=0.3, top_p=0.85)
        match = re.search(r'\{[^}]+\}', result)
        if match:
            data = json.loads(match.group())
            verdict = data.get("verdict", "safe").lower()
            if verdict not in ["safe", "flagged", "blocked"]:
                verdict = "safe"
            return {
                "verdict": verdict,
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                "labels": data.get("labels", []) if isinstance(data.get("labels"), list) else [],
                "reason": data.get("reason", "Safe content"),
                "raw_label": data.get("labels", [""])[0] if data.get("labels") else "",
            }
    except Exception as e:
        print(f"Moderation error: {e}")
    
    return {"verdict": "safe", "confidence": 0.0, "labels": [], "reason": "Safe content", "raw_label": ""}

@app.post("/api/moderate")
async def moderate(req: AnalyzePostRequest):
    result = await analyze_post(req)
    mod = result["moderation"]
    mod["review_flag"] = result["review_flag"]
    return mod

# ================== CONVERSATIONAL SUMMARIZATION ==================
@app.post("/api/summarize-thread")
async def summarize_thread(req: SummarizeThreadRequest):
    """
    Casual, conversational thread summaries
    """
    texts = [clean(t) for t in req.texts if t]
    if not texts:
        return {"summary": ""}

    main_post = texts[0]
    replies = texts[1:] if len(texts) > 1 else []
    reply_count = len(replies)
    
    context_parts = []
    context_parts.append(f"Main Post: {main_post}")
    
    for i, reply in enumerate(replies[:20], 1):
        context_parts.append(f"Reply {i}: {reply}")
    
    full_context = "\n\n".join(context_parts)
    
    if req.image_counts:
        img_count = sum(max(0, c) for c in req.image_counts)
        if img_count > 0:
            full_context += f"\n\n[Thread has {img_count} image{'s' if img_count != 1 else ''}]"
    
    # CONVERSATIONAL PROMPTS
    if reply_count == 0:
        if len(main_post) <= 200:
            return {"summary": main_post}
        
        prompt = f"""Summarize this post in simple, everyday language. Write like you're explaining it to a friend. Use casual words and keep it natural.

Post:
{main_post}

Casual summary:"""
        max_chars = 250
        
    elif reply_count <= 5:
        prompt = f"""Summarize this conversation in a casual, easy to read way. Write naturally like you're telling someone what happened. You can use bullet points if it makes sense.

What to cover:
- What's the main topic
- Key points people made
- Any interesting takes

Thread:
{full_context}

Summary:"""
        max_chars = 350
        
    else:
        prompt = f"""Give me a clear summary of this discussion. Write it naturally and conversationally, like you're explaining it to someone. You can break it into paragraphs or use bullet points if that helps.

Include:
- What the discussion is about
- Different opinions people shared
- Where people agreed or disagreed
- Any interesting conclusions

Thread:
{full_context[:4000]}

Summary:"""
        max_chars = 500
    
    try:
        print(f"Generating conversational summary for thread with {reply_count} replies...")
        
        summary = await generate(
            prompt, 
            max_new=300,
            temperature=0.75,  # Slightly higher for more natural language
            top_p=0.9,
            char_limit=max_chars
        )
        
        summary = clean(summary)
        
        # Remove formal markers
        summary = re.sub(r'^(here is |here\'s |the |this )?summar[yi](?: of| is)?:?\s*', '', summary, flags=re.I)
        summary = re.sub(r'^(the thread|the discussion|the post|the conversation)\s+(discusses?|is about|covers?|explores?|examines?)\s*', '', summary, flags=re.I)
        summary = re.sub(r'^(in summary|to summarize|in conclusion),?\s*', '', summary, flags=re.I)
        
        # Keep markdown for readability but remove bold/italic
        summary = re.sub(r'\*\*([^*]+)\*\*', r'\1', summary)
        summary = re.sub(r'\*([^*]+)\*', r'\1', summary)
        
        # Remove hyphens at start of lines (but keep mid-sentence hyphens)
        summary = re.sub(r'^\s*[-•]\s*', '', summary, flags=re.MULTILINE)
        
        summary = clean(summary)
        
        if len(summary) > max_chars * 1.2:
            summary = smart_truncate(summary, max_chars, prefer_complete=True)
        
        if not summary or len(summary) < 20:
            if reply_count == 0:
                summary = smart_truncate(main_post, 200, prefer_complete=False)
            else:
                summary = f"{smart_truncate(main_post, 120, prefer_complete=False)} There are {reply_count} {'reply' if reply_count == 1 else 'replies'} discussing this."
        
        print(f"✓ Generated summary: {len(summary)} chars")
        return {"summary": summary}
        
    except Exception as e:
        print(f"Summary error: {e}")
        if reply_count == 0:
            return {"summary": smart_truncate(main_post, 200, prefer_complete=False)}
        else:
            fallback = f"{smart_truncate(main_post, 150, prefer_complete=False)} {reply_count} people replied with different takes."
            return {"summary": fallback}

# ================== CONVERSATIONAL CONDENSE ==================
@app.post("/api/condense-to-post")
async def condense(data: Dict[str, Any]):
    """Make text shorter while keeping it casual and natural"""
    text = clean(data.get("text", ""))
    if not text:
        return {"draft": ""}

    if len(text) <= MAX_POST_CHARS:
        return {"draft": text}
    
    word_count = len(text.split())
    char_count = len(text)
    reduction_ratio = MAX_POST_CHARS / char_count
    target_words = max(int(word_count * reduction_ratio * 0.9), 40)

    prompt = f"""Make this text shorter (around {target_words} words) but keep it casual and easy to read. Write naturally like you would in a social media post.

Important:
- Keep all the key facts and details
- Use simple everyday words, not fancy vocabulary
- Sound natural and conversational
- Must fit in {MAX_POST_CHARS} characters
- No emojis, hashtags, or bullet points
- Just write it out normally

Original:
{text}

Shorter version:"""
    
    try:
        print(f"Making text shorter: {char_count} → {MAX_POST_CHARS} chars")
        
        draft = await generate(
            prompt,
            max_new=500,
            temperature=0.8,
            top_p=0.9,
            char_limit=MAX_POST_CHARS
        )
        
        draft = remove_emojis_and_symbols(draft).strip()
        draft = re.sub(r'#\w+', '', draft)
        draft = clean(draft)
        
        # Remove meta-commentary
        draft = re.sub(r'^(shorter version|condensed|here is|here\'s):?\s*', '', draft, flags=re.I).strip()
        
        if len(draft) > MAX_POST_CHARS * 1.15:
            print(f"⚠ Over limit ({len(draft)} chars), truncating...")
            draft = smart_truncate(draft, MAX_POST_CHARS, prefer_complete=True)
        
        print(f"✓ Made shorter: {len(draft)} chars")
        return {"draft": draft}
        
    except Exception as e:
        print(f"Condense error: {e}")
        return {"draft": smart_truncate(text, MAX_POST_CHARS, prefer_complete=False)}


@app.post("/api/sentiment-only")
async def sentiment_only(data: Dict[str, Any]):
    """Fast sentiment analysis only - no moderation"""
    text = clean(data.get("text", ""))
    if not text:
        return {"sentiment": {"label": "NEUTRAL", "confidence": 0.0}}
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: sentiment_pipe(text[:512], batch_size=1)[0]
        )
        return {"sentiment": {"label": result["label"], "confidence": float(result["score"])}}
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"sentiment": {"label": "NEUTRAL", "confidence": 0.0}}

# ================== CONVERSATIONAL REWRITE ==================
@app.post("/api/draft-post")
async def draft_post(data: Dict[str, Any]):
    """Rewrite text naturally and conversationally"""
    prompt_text = clean(data.get("prompt", ""))
    if len(prompt_text) < 3:
        return {"draft": "", "error": "Prompt too short"}

    char_target = MAX_POST_CHARS if len(prompt_text) <= MAX_POST_CHARS else int(MAX_POST_CHARS * 0.95)

    prompt = f"""Rewrite this text in your own words. Keep it casual and natural, like you're posting on social media. Don't make it formal or use fancy words.

Important:
- Keep the same meaning and all important details
- Use simple everyday language
- Sound natural and conversational
- Stay under {char_target} characters
- No emojis, hashtags, or bullet points
- Just write it naturally

Original:
{prompt_text}

Rewritten:"""
    
    try:
        print(f"Rewriting text: {len(prompt_text)} chars")
        
        draft = await generate(
            prompt,
            max_new=500,
            temperature=0.85,  # Higher for creative, casual rephrasing
            top_p=0.9,
            char_limit=char_target
        )
        
        draft = remove_emojis_and_symbols(draft)
        draft = re.sub(r'#\w+', '', draft)
        draft = clean(draft)
        
        # Remove meta-commentary
        draft = re.sub(r'^(rewritten|here is|here\'s):?\s*', '', draft, flags=re.I).strip()
        
        if len(draft) > MAX_POST_CHARS * 1.15:
            print(f"⚠ Over limit ({len(draft)} chars), truncating...")
            draft = smart_truncate(draft, MAX_POST_CHARS, prefer_complete=True)
        
        print(f"✓ Rewritten: {len(draft)} chars")
        return {"draft": draft}
        
    except Exception as e:
        print(f"Draft error: {e}")
        return {"draft": smart_truncate(prompt_text, MAX_POST_CHARS, prefer_complete=False)}
    

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        workers=1,
        log_level="info"
    )
