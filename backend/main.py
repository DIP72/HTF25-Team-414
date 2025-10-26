# backend/main.py - BALANCED AI MODERATION
"""
AI-Powered Threads Backend
Context-aware content moderation using Qwen LLM
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import uvicorn

# ================== CONFIG ==================
MAX_POST_CHARS = 1000
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_WORKERS = 2
MIN_LENGTH_FOR_AI = 100

# ================== GLOBAL RESOURCES ==================
HAS_CUDA = torch.cuda.is_available()
DTYPE = torch.bfloat16 if HAS_CUDA and torch.cuda.is_bf16_supported() else torch.float32
DEVICE = "cuda" if HAS_CUDA else "cpu"

tokenizer = None
model = None
sentiment_pipe = None
executor = None

# ================== HELPERS ==================
def clean(text: str) -> str:
    """Basic text cleaning"""
    return re.sub(r'\s+', ' ', (text or "")).strip()

def remove_emojis_and_symbols(s: str) -> str:
    """Remove emojis and special symbols"""
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
    """Smart truncation at sentence boundaries"""
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
            if last_pos >= limit * 0.7:
                result = truncated[:last_pos].strip()
                if not result.endswith(('.', '!', '?')):
                    result += '.'
                return result
    
    truncated = text[:limit]
    sentence_ends = [truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?')]
    last_sentence = max(sentence_ends)
    
    if last_sentence != -1 and last_sentence >= int(limit * 0.6):
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

def clean_generated_text(text: str, stop_sequences: List[str] = None) -> str:
    """Advanced cleaning to remove unwanted patterns and repetition"""
    text = clean(text)
    
    if not text:
        return ""
    
    stop_patterns = [
        r"\n\n+",
        r"^(original|post|thread|summary|rewritten|shorter|version|update|details|note|here|example)[:]\s*",
        r"(here is|here's|this is|according to|based on|in conclusion)\s+(the|a|an)\s+",
    ]
    
    for pattern in stop_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
    
    if stop_sequences:
        earliest_pos = len(text)
        for stop in stop_sequences:
            pos = text.lower().find(stop.lower())
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        if earliest_pos < len(text):
            text = text[:earliest_pos].strip()
    
    text = re.sub(
        r'^(here is |here\'s |the |this |that |according to |based on |in |as |to )*'
        r'(summary|summar|condensed|rewritten|shortened|shorter|brief|quick|casual|'
        r'version|draft|response|answer|result|output|text|post|update|details?)[\s:,;-]*',
        '', text, flags=re.IGNORECASE
    )
    
    if len(text) > 100 and not text.endswith(('.', '!', '?', '"', "'")):
        last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_punct > len(text) * 0.65:
            text = text[:last_punct + 1]
    
    text = re.sub(r'[,;:\s]+$', '', text)
    
    return text.strip()

def check_similarity(text1: str, text2: str) -> float:
    """Simple word overlap similarity check"""
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

# ================== GENERATION ==================
def generate_sync(prompt: str, max_new: int = 256, temperature: float = 0.7, 
                 top_p: float = 0.9, char_limit: Optional[int] = None, 
                 stop_sequences: List[str] = None) -> str:
    global model, tokenizer
    if not model or not tokenizer:
        return ""
    try:
        avg_chars_per_token = 4.0
        if char_limit:
            target_new_tokens = min(int(char_limit / avg_chars_per_token * 1.2), max_new)
        else:
            target_new_tokens = max_new
        
        target_new_tokens = max(30, min(target_new_tokens, 256))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if HAS_CUDA:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        stop_token_ids = []
        if stop_sequences:
            for seq in stop_sequences:
                tokens = tokenizer.encode(seq, add_special_tokens=False)
                if tokens:
                    stop_token_ids.extend(tokens)

        gen_kwargs = {
            "max_new_tokens": target_new_tokens,
            "min_new_tokens": 15,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 40,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "encoder_repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "diversity_penalty": 0.5,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": [tokenizer.eos_token_id] + stop_token_ids if stop_token_ids else tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1,
        }

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        generated = clean_generated_text(generated, stop_sequences)
        return generated
    except Exception as e:
        print(f"Generation error: {e}")
        return ""

async def generate(prompt: str, max_new: int = 256, temperature: float = 0.7,
                   top_p: float = 0.9, char_limit: Optional[int] = None, 
                   stop_sequences: List[str] = None) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, generate_sync, prompt, max_new, temperature, top_p, char_limit, stop_sequences
    )

# ================== MODEL LOADING ==================
def load_models():
    global tokenizer, model, sentiment_pipe, executor
    print(f"🚀 Loading {MODEL_NAME}...")
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    if not HAS_CUDA:
        model = model.to(DTYPE)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    print(f"✅ Model loaded on {DEVICE.upper()}")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if HAS_CUDA else -1,
        batch_size=1,
        max_length=256,
        truncation=True
    )
    print(f"✅ Ready on {DEVICE.upper()}")

def cleanup_models():
    global executor
    if executor:
        executor.shutdown(wait=True)
    print("🛑 Shutdown complete")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    cleanup_models()

# ================== FASTAPI SETUP ==================
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ================== SENTIMENT + CONTEXT-AWARE MODERATION ==================
@app.post("/api/analyze-post")
async def analyze_post(req: AnalyzePostRequest):
    text = clean(req.text)
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "confidence": 0.0},
            "moderation": {"verdict": "safe", "confidence": 0.0, "labels": [], "reason": "Empty", "raw_label": ""},
            "review_flag": False
        }
    
    sentiment_task = asyncio.create_task(_get_sentiment(text))
    moderation_task = asyncio.create_task(_get_moderation_ai(text))
    sentiment, moderation = await asyncio.gather(sentiment_task, moderation_task)

    return {
        "sentiment": sentiment,
        "moderation": moderation,
        "review_flag": moderation["verdict"] in ["flagged", "blocked"]
    }

async def _get_sentiment(text: str) -> Dict[str, Any]:
    """Enhanced sentiment analysis with profanity detection"""
    try:
        text_lower = text.lower()
        
        # Only standalone profanity (single words)
        standalone_profanity = [
            'fuck', 'shit', 'bitch', 'damn', 'bastard', 'crap', 'piss',
            'dick', 'cock', 'pussy', 'slut', 'whore'
        ]
        
        # Check if text is ONLY profanity (single word or very short)
        words = text.split()
        if len(words) <= 2:
            for word in standalone_profanity:
                if re.search(rf'\b{re.escape(word)}\b', text_lower):
                    return {"label": "NEGATIVE", "confidence": 0.95}
        
        # Run normal sentiment analysis for longer text
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: sentiment_pipe(text[:256], batch_size=1)[0]
        )
        
        label = result["label"].upper()
        confidence = float(result["score"])
        
        return {"label": label, "confidence": confidence}
        
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"label": "NEUTRAL", "confidence": 0.5}

async def _get_moderation_ai(text: str) -> Dict[str, Any]:
    """Context-aware AI moderation - blocks only inappropriate use"""
    
    text_lower = text.lower().strip()
    word_count = len(text.split())
    
    # ONLY block standalone profanity (single words or very short)
    if word_count <= 3:
        standalone_blocked = [
            'fuck', 'shit', 'bitch', 'damn', 'bastard', 'ass', 'crap',
            'dick', 'cock', 'pussy', 'slut', 'whore', 'cunt', 'fag',
            'fuk', 'fck', 'sht', 'btch', 'a$$', 'f*ck', 'sh!t'
        ]
        
        for keyword in standalone_blocked:
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                print(f"🚫 BLOCKED standalone profanity: {keyword}")
                return {
                    "verdict": "blocked",
                    "confidence": 0.98,
                    "labels": ["profanity"],
                    "reason": f"Contains inappropriate language",
                    "raw_label": "profanity",
                }
    
    # ONLY block standalone sexual terms (single words)
    if word_count <= 2:
        sexual_standalone = ['sex', 'porn', 'nsfw', 'nude', 'naked', 'xxx']
        for keyword in sexual_standalone:
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                print(f"🚫 BLOCKED standalone sexual: {keyword}")
                return {
                    "verdict": "blocked",
                    "confidence": 0.98,
                    "labels": ["sexual"],
                    "reason": f"Contains inappropriate sexual content",
                    "raw_label": "sexual",
                }
    
    # For longer text, use AI with context-awareness
    prompt = f"""You are a content moderation AI with CONTEXT AWARENESS. Analyze if this text is inappropriate.

IMPORTANT RULES:
- NEWS REPORTING is SAFE (accidents, crimes, disasters reported factually)
- EDUCATIONAL/INFORMATIONAL content is SAFE  
- Legitimate discussions about serious topics are SAFE
- ONLY flag if the text is:
  1. Direct harassment or hate speech
  2. Explicit sexual content (not just mentioning words in context)
  3. Promoting violence or illegal activities
  4. Spam or scam content

CATEGORIES:
1. SAFE - News, education, legitimate discussion
2. PROFANITY - Gratuitous profanity without purpose
3. HATE_SPEECH - Targeting groups with hate
4. VIOLENCE - Promoting or celebrating violence
5. HARASSMENT - Direct attacks on individuals
6. SEXUAL - Explicit sexual content
7. SPAM - Advertising or scams

Text to analyze:
"{text}"

Respond in EXACTLY this JSON format:
{{"category": "SAFE/PROFANITY/HATE_SPEECH/VIOLENCE/HARASSMENT/SEXUAL/SPAM", "confidence": 0.0-1.0, "reason": "brief explanation"}}

JSON:"""
    
    try:
        print(f"🔍 AI Moderating: {text[:100]}...")
        
        result = await generate(
            prompt,
            max_new=80,
            temperature=0.3,  # Low but not too strict
            top_p=0.85,
            stop_sequences=["Text:", "Respond:", "\n\n"]
        )
        
        result = result.strip()
        match = re.search(r'\{[^}]+\}', result)
        
        if match:
            data = json.loads(match.group())
            category = data.get("category", "SAFE").upper()
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            reason = data.get("reason", "Content analyzed")
            
            # Only block clear violations
            if category == "SAFE":
                verdict = "safe"
                labels = []
            else:
                verdict = "blocked"
                labels = [category.lower().replace("_", "-")]
            
            print(f"✓ Moderation: {verdict} ({category}, {confidence:.2f}) - {reason}")
            
            return {
                "verdict": verdict,
                "confidence": confidence,
                "labels": labels,
                "reason": reason,
                "raw_label": category.lower(),
            }
        else:
            print(f"⚠️ Failed to parse AI result, defaulting to safe")
            return {
                "verdict": "safe",
                "confidence": 0.7,
                "labels": [],
                "reason": "Could not analyze",
                "raw_label": "",
            }
    
    except Exception as e:
        print(f"AI Moderation error: {e}")
        return {
            "verdict": "safe",
            "confidence": 0.5,
            "labels": [],
            "reason": "Analysis failed",
            "raw_label": "",
        }

@app.post("/api/moderate")
async def moderate(req: AnalyzePostRequest):
    result = await analyze_post(req)
    return result["moderation"]

# ================== THREAD SUMMARIZATION ==================
@app.post("/api/summarize-thread")
async def summarize_thread(req: SummarizeThreadRequest):
    texts = [clean(t) for t in req.texts if t]
    if not texts:
        return {"summary": ""}
    
    main_post = texts[0]
    replies = texts[1:] if len(texts) > 1 else []
    reply_count = len(replies)
    
    if reply_count == 0 and len(main_post) <= 200:
        return {"summary": main_post}
    if reply_count == 0 and len(main_post) <= 300:
        return {"summary": smart_truncate(main_post, 250, prefer_complete=True)}
    
    context_parts = [f"Main: {main_post[:500]}"]
    for i, reply in enumerate(replies[:6], 1):
        context_parts.append(f"Reply {i}: {reply[:150]}")
    context = "\n".join(context_parts)
    max_chars = 400
    
    if req.image_counts:
        img_count = sum(max(0, c) for c in req.image_counts)
        if img_count > 0:
            context += f"\n[{img_count} image{'s' if img_count != 1 else ''}]"
    
    prompt = f"""Read this discussion and write a completely NEW summary in 2-4 sentences using YOUR OWN words. Don't copy the text.

Discussion:
{context}

Your summary (own words):"""
    
    try:
        summary = await generate(prompt, max_new=150, temperature=0.85, top_p=0.92, char_limit=max_chars, stop_sequences=["Post:", "Reply:", "Main:", "Discussion:", "Original:", "\n\n"])
        summary = clean_generated_text(summary, ["Post:", "Reply:", "Thread:"])
        similarity = check_similarity(summary, main_post[:200])
        if similarity > 0.8:
            summary = smart_truncate(main_post, max_chars, prefer_complete=True)
        if len(summary) > max_chars * 1.3:
            summary = smart_truncate(summary, max_chars, prefer_complete=True)
        if not summary or len(summary) < 30:
            summary = f"{smart_truncate(main_post, 150, prefer_complete=False)} ({reply_count} replies)"
        return {"summary": summary}
    except Exception as e:
        print(f"Summary error: {e}")
        return {"summary": smart_truncate(main_post, 150, prefer_complete=False)}

# ================== CONDENSE TO POST ==================
@app.post("/api/condense-to-post")
async def condense(data: Dict[str, Any]):
    text = clean(data.get("text", ""))
    if not text:
        return {"draft": ""}
    if len(text) <= MAX_POST_CHARS:
        return {"draft": text}
    if len(text) <= MAX_POST_CHARS + 50:
        return {"draft": smart_truncate(text, MAX_POST_CHARS, prefer_complete=True)}
    if len(text) > 2500:
        text = text[:2500]
    
    target_chars = int(MAX_POST_CHARS * 0.88)
    word_count = len(text.split())
    target_words = int(word_count * (target_chars / len(text)))
    
    prompt = f"""Rewrite this in about {target_words} words ({target_chars} characters). Keep ALL key facts but make it shorter and clearer.

Original text:
{text}

Shorter version (keep facts, fewer words):"""
    
    try:
        draft = await generate(prompt, max_new=180, temperature=0.75, top_p=0.9, char_limit=target_chars, stop_sequences=["Original:", "Text:", "\n\n\n"])
        draft = remove_emojis_and_symbols(draft)
        draft = re.sub(r'#\w+', '', draft)
        draft = clean_generated_text(draft)
        similarity = check_similarity(draft, text[:300])
        if similarity > 0.9:
            draft = smart_truncate(text, MAX_POST_CHARS, prefer_complete=True)
        if len(draft) > MAX_POST_CHARS * 1.1:
            draft = smart_truncate(draft, MAX_POST_CHARS, prefer_complete=True)
        if not draft or len(draft) < 50:
            draft = smart_truncate(text, MAX_POST_CHARS, prefer_complete=False)
        return {"draft": draft}
    except Exception as e:
        print(f"Condense error: {e}")
        return {"draft": smart_truncate(text, MAX_POST_CHARS, prefer_complete=False)}

# ================== SENTIMENT ONLY ==================
@app.post("/api/sentiment-only")
async def sentiment_only(data: Dict[str, Any]):
    text = clean(data.get("text", ""))
    if not text:
        return {"sentiment": {"label": "NEUTRAL", "confidence": 0.0}}
    
    sentiment = await _get_sentiment(text)
    return {"sentiment": sentiment}

# ================== DRAFT POST (REWRITE) ==================
@app.post("/api/draft-post")
async def draft_post(data: Dict[str, Any]):
    prompt_text = clean(data.get("prompt", ""))
    if len(prompt_text) < 3:
        return {"draft": "", "error": "Prompt too short"}
    if len(prompt_text) <= MAX_POST_CHARS:
        return {"draft": remove_emojis_and_symbols(prompt_text)}
    if len(prompt_text) > 2000:
        prompt_text = prompt_text[:2000]
    
    char_target = int(MAX_POST_CHARS * 0.88)
    
    prompt = f"""Rewrite this in a casual, natural way. Use simple words and keep it under {char_target} characters. Don't copy the original wording.

Original:
{prompt_text}

Casual rewrite (different words):"""
    
    try:
        draft = await generate(prompt, max_new=180, temperature=0.88, top_p=0.92, char_limit=char_target, stop_sequences=["Original:", "Text:", "\n\n\n"])
        draft = remove_emojis_and_symbols(draft)
        draft = re.sub(r'#\w+', '', draft)
        draft = clean_generated_text(draft)
        similarity = check_similarity(draft, prompt_text[:300])
        if similarity > 0.85:
            draft = smart_truncate(prompt_text, MAX_POST_CHARS, prefer_complete=True)
        if len(draft) > MAX_POST_CHARS * 1.1:
            draft = smart_truncate(draft, MAX_POST_CHARS, prefer_complete=True)
        if not draft or len(draft) < 50:
            draft = smart_truncate(prompt_text, MAX_POST_CHARS, prefer_complete=False)
        return {"draft": draft}
    except Exception as e:
        print(f"Draft error: {e}")
        return {"draft": smart_truncate(prompt_text, MAX_POST_CHARS, prefer_complete=False)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
