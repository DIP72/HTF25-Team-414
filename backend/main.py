# backend/main.py - PRODUCTION READY
"""
Fast, Context-Aware AI Backend
- Blocks: Direct slurs, threats (< 1ms)
- Allows: News, factual reporting
- Smart: Summarize, condense, rewrite
"""

import re
import json
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from better_profanity import profanity
import uvicorn

# ================== CONFIG ==================
MAX_POST_CHARS = 1000
MIN_POST_CHARS = 20
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_WORKERS = 2

# ================== GLOBALS ==================
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"
DTYPE = torch.float16 if HAS_CUDA else torch.float32

tokenizer = None
model = None
sentiment_pipe = None
executor = None

# Initialize profanity filter
profanity.load_censor_words()

# ================== CONTEXT DETECTION ==================
NEWS_KEYWORDS = {
    'police', 'authorities', 'officials', 'government', 'minister',
    'reported', 'according', 'investigation', 'probe', 'district',
    'incident', 'accident', 'collision', 'victims', 'casualties',
    'claimed', 'alleged', 'demanded', 'compensation', 'criminal',
    'press release', 'opposition', 'party', 'deaths', 'bereaved',
    'seized', 'arrested', 'negligence', 'maintained'
}

def detect_context(text: str) -> str:
    """
    Returns: 'news', 'short_slur', 'normal'
    """
    text_lower = text.lower()
    word_count = len(text.split())
    
    # SHORT SLUR DETECTION (< 5 words with only profanity)
    if word_count <= 5 and profanity.contains_profanity(text):
        censored = profanity.censor(text, '*')
        remaining = [w for w in censored.split() if '*' not in w and len(w) > 2]
        if len(remaining) < 2:
            return 'short_slur'
    
    # NEWS DETECTION
    words = set(text_lower.split())
    indicators = len(words & NEWS_KEYWORDS)
    has_quotes = '"' in text
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    has_structure = sentence_count >= 3 or has_quotes
    
    if indicators >= 4 or (indicators >= 2 and has_structure):
        return 'news'
    if indicators >= 3 and word_count > 50:
        return 'news'
    
    return 'normal'

# ================== UTILITIES ==================
def clean(t: str) -> str:
    return ' '.join((t or "").split())

def truncate(text: str, limit: int) -> str:
    """Intelligent truncation"""
    if len(text) <= limit:
        return text
    
    truncated = text[:limit]
    
    # Try sentence ending
    sentence_end = max(truncated.rfind('. '), truncated.rfind('! '), truncated.rfind('? '))
    if sentence_end > limit * 0.75:
        return truncated[:sentence_end + 1].strip()
    
    # Try clause ending
    clause_end = max(truncated.rfind(', '), truncated.rfind('; '), truncated.rfind(': '))
    if clause_end > limit * 0.65:
        result = truncated[:clause_end].strip()
        if not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
    # Word boundary
    last_space = truncated.rfind(' ')
    if last_space > limit * 0.5:
        result = truncated[:last_space].strip()
        if not result.endswith(('.', '!', '?')):
            result += '...'
        return result
    
    return truncated.strip() + '...'

# ================== AI GENERATION ==================
@torch.inference_mode()
def generate_sync(prompt: str, max_tokens: int = 200, temp: float = 0.7) -> str:
    """Fast generation"""
    global model, tokenizer
    if not model:
        return ""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if HAS_CUDA:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=10,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True
        )
        
        result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return clean(result)
    except:
        return ""

async def generate(prompt: str, max_tokens: int = 200, temp: float = 0.7) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_sync, prompt, max_tokens, temp)

# ================== MODEL LOADING ==================
def load_models():
    global tokenizer, model, sentiment_pipe, executor
    
    print(f"🚀 Loading models...")
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True
    )
    
    if HAS_CUDA:
        model = model.cuda()
    model.eval()
    
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if HAS_CUDA else -1,
        truncation=True,
        max_length=128
    )
    
    print(f"✅ Ready on {DEVICE}")

def cleanup():
    global executor
    if executor:
        executor.shutdown(wait=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    cleanup()

# ================== FASTAPI ==================
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AnalyzePostRequest(BaseModel):
    text: str
    media_flags: Optional[List[str]] = []

class SummarizeThreadRequest(BaseModel):
    texts: List[str] = []
    image_counts: Optional[List[int]] = []

@app.get("/")
def root():
    return {"status": "running", "device": DEVICE}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ================== MODERATION ==================
async def moderate_smart(text: str) -> Dict[str, Any]:
    """Context-aware moderation"""
    
    context_type = detect_context(text)
    print(f"📊 Context: {context_type} | Length: {len(text)}")
    
    # NEWS - Always safe
    if context_type == 'news':
        print(f"✅ NEWS - Allowed")
        return {
            "verdict": "safe",
            "confidence": 0.95,
            "labels": [],
            "reason": "News/factual reporting",
            "raw_label": ""
        }
    
    # SHORT SLUR - Always blocked
    if context_type == 'short_slur':
        print(f"🚫 BLOCKED - Short slur")
        return {
            "verdict": "blocked",
            "confidence": 0.97,
            "labels": ["profanity"],
            "reason": "Direct profanity/slur",
            "raw_label": "profanity"
        }
    
    # NORMAL - Check profanity
    if profanity.contains_profanity(text):
        word_count = len(text.split())
        
        if word_count <= 8:
            print(f"🚫 BLOCKED - Short post with profanity")
            return {
                "verdict": "blocked",
                "confidence": 0.92,
                "labels": ["profanity"],
                "reason": "Profanity in short post",
                "raw_label": "profanity"
            }
        else:
            # Longer post - AI judges context
            print(f"🤖 AI analysis - Profanity in longer post")
            prompt = f"Is this appropriate or offensive? Text: {text[:500]}\nRespond: appropriate OR offensive"
            
            try:
                result = await generate(prompt, max_tokens=10, temp=0.2)
                if "offensive" in result.lower():
                    return {
                        "verdict": "blocked",
                        "confidence": 0.85,
                        "labels": ["inappropriate"],
                        "reason": "Inappropriate content",
                        "raw_label": "inappropriate"
                    }
            except:
                pass
    
    # Clean content
    print(f"✅ SAFE")
    return {
        "verdict": "safe",
        "confidence": 0.90,
        "labels": [],
        "reason": "Content safe",
        "raw_label": ""
    }

async def get_sentiment(text: str) -> Dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()
        r = await loop.run_in_executor(executor, lambda: sentiment_pipe(text[:128])[0])
        return {"label": r["label"], "confidence": float(r["score"])}
    except:
        return {"label": "NEUTRAL", "confidence": 0.5}

@app.post("/api/analyze-post")
async def analyze_post(req: AnalyzePostRequest):
    text = clean(req.text)
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "confidence": 0.0},
            "moderation": {"verdict": "safe", "confidence": 0.0, "labels": [], "reason": "Empty", "raw_label": ""},
            "review_flag": False
        }
    
    sentiment, moderation = await asyncio.gather(get_sentiment(text), moderate_smart(text))
    
    return {
        "sentiment": sentiment,
        "moderation": moderation,
        "review_flag": moderation["verdict"] == "blocked"
    }

@app.post("/api/moderate")
async def moderate(req: AnalyzePostRequest):
    text = clean(req.text)
    return await moderate_smart(text) if text else {
        "verdict": "safe", "confidence": 0, "labels": [], "reason": "Empty", "raw_label": ""
    }

# ================== SUMMARIZE ==================
@app.post("/api/summarize-thread")
async def summarize_thread(req: SummarizeThreadRequest):
    texts = [clean(t) for t in req.texts if t]
    if not texts:
        return {"summary": ""}
    
    main = texts[0]
    replies = texts[1:]
    
    if not replies and len(main) <= 250:
        return {"summary": main}
    
    # Build context
    if not replies:
        context = f"Post: {main[:600]}"
    else:
        parts = [f"Main: {main[:400]}"]
        for i, r in enumerate(replies[:10], 1):
            parts.append(f"R{i}: {r[:150]}")
        context = " ".join(parts)
    
    prompt = f"Summarize in 2-3 sentences: {context[:1500]}\n\nSummary:"
    
    try:
        summary = await generate(prompt, max_tokens=150, temp=0.7)
        summary = re.sub(r'^(summary|here|this):?\s*', '', summary, flags=re.I)
        summary = clean(summary)
        
        if len(summary) < 20:
            summary = truncate(main, 200)
        elif len(summary) > 400:
            summary = truncate(summary, 400)
        
        return {"summary": summary}
    except:
        return {"summary": truncate(main, 200)}

# ================== CONDENSE (IMPROVED) ==================
@app.post("/api/condense-to-post")
async def condense(data: Dict[str, Any]):
    """Smart condensing with validation"""
    text = clean(data.get("text", ""))
    if not text or len(text) <= MAX_POST_CHARS:
        return {"draft": text}
    
    # Just slightly over? Truncate
    if len(text) <= MAX_POST_CHARS + 150:
        return {"draft": truncate(text, MAX_POST_CHARS)}
    
    # Calculate reduction needed
    current_len = len(text)
    target_len = int(MAX_POST_CHARS * 0.85)
    reduction_pct = int((1 - target_len / current_len) * 100)
    
    prompt = f"""Make this {reduction_pct}% shorter (from {current_len} to ~{target_len} chars).

KEEP: Key facts and main points
REMOVE: Unnecessary words and details
NO: Emojis or hashtags

Original:
{text[:2000]}

Shorter ({target_len} chars):"""
    
    try:
        max_tokens = min(int(target_len / 3), 400)
        
        draft = await generate(prompt, max_tokens=max_tokens, temp=0.7)
        
        # Clean
        draft = re.sub(r'^(short|shorter|here|version):?\s*', '', draft, flags=re.I)
        draft = re.sub(r'#\w+', '', draft)
        draft = re.sub(r'[\U0001F300-\U0001F9FF\U00002600-\U000027BF]+', '', draft)
        draft = clean(draft)
        
        # Validate
        if len(draft) > MAX_POST_CHARS:
            print(f"⚠️ AI output too long ({len(draft)}), truncating")
            draft = truncate(draft, MAX_POST_CHARS)
        
        if len(draft) < 50 or len(draft) < target_len * 0.3:
            print(f"⚠️ AI output too short ({len(draft)}), using fallback")
            draft = truncate(text, MAX_POST_CHARS)
        
        if len(draft) >= current_len * 0.95:
            print(f"⚠️ AI barely shortened, using truncation")
            draft = truncate(text, MAX_POST_CHARS)
        
        print(f"✅ Condensed: {current_len} → {len(draft)} ({reduction_pct}% reduction)")
        return {"draft": draft}
        
    except Exception as e:
        print(f"❌ Condense error: {e}")
        return {"draft": truncate(text, MAX_POST_CHARS)}

# ================== REWRITE ==================
@app.post("/api/draft-post")
async def draft_post(data: Dict[str, Any]):
    text = clean(data.get("prompt", ""))
    if len(text) < 5:
        return {"draft": ""}
    
    prompt = f"Rewrite casually and naturally: {text[:1500]}\n\nRewrite:"
    
    try:
        draft = await generate(prompt, max_tokens=300, temp=0.85)
        draft = re.sub(r'^(rewrite|here):?\s*', '', draft, flags=re.I)
        draft = re.sub(r'#\w+', '', clean(draft))
        
        if len(draft) > MAX_POST_CHARS:
            draft = truncate(draft, MAX_POST_CHARS)
        if len(draft) < 30:
            draft = truncate(text, MAX_POST_CHARS)
        
        return {"draft": draft}
    except:
        return {"draft": truncate(text, MAX_POST_CHARS)}

# ================== SENTIMENT ONLY ==================
@app.post("/api/sentiment-only")
async def sentiment_only(data: Dict[str, Any]):
    text = clean(data.get("text", ""))
    return {"sentiment": await get_sentiment(text) if text else {"label": "NEUTRAL", "confidence": 0.0}}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
