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

MAXPOSTCHARS = 1000
MODELNAME = "Qwen/Qwen2.5-1.5B-Instruct"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HASCUDA = torch.cuda.is_available()
DTYPE = torch.bfloat16 if HASCUDA and torch.cuda.is_bf16_supported() else torch.float16
DEVICE = 0 if HASCUDA else -1

executor = ThreadPoolExecutor(max_workers=4)

tokenizer = None
model = None
sentiment_pipe = None

def load():
    global tokenizer, model, sentiment_pipe
    print(f"🚀 Loading {MODELNAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODELNAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELNAME,
        trust_remote_code=True,
        device_map="auto" if HASCUDA else None,
        dtype=DTYPE,
    )
    model.eval()
    
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=DEVICE
    )
    
    print("✅ Models loaded successfully")

load()

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()

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

def trim_to_sentence_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    
    truncated = text[:limit]
    last_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    
    if last_end != -1 and last_end > int(limit * 0.5):
        result = truncated[:last_end+1].strip()
    else:
        result = truncated.rsplit(' ', 1)[0].strip()
    
    if not result:
        result = truncated[:limit]
    
    if not result.endswith(('.', '!', '?')):
        result = result.rstrip('- ') + '.'
    
    return result

def generate(prompt: str, max_new: int = 256, use_sampling: bool = False, char_limit: Optional[int] = None) -> str:
    global model, tokenizer
    if not model or not tokenizer:
        return ""
    
    avg_chars_per_token = 4.0
    if char_limit is None:
        target_chars = MAXPOSTCHARS
    else:
        target_chars = char_limit
    
    target_new_tokens = max(8, int(math.ceil(target_chars / avg_chars_per_token)))
    target_new_tokens = min(target_new_tokens, max_new)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32000)
    if HASCUDA:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    input_len = inputs["input_ids"].shape[1]
    model_max = getattr(tokenizer, "model_max_length", None) or getattr(model.config, "n_positions", None) or 32000
    safe_max_new = max(8, min(target_new_tokens, max(8, model_max - input_len - 8)))
    
    gen_kwargs = {
        "max_new_tokens": int(safe_max_new),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if use_sampling:
        gen_kwargs.update({"do_sample": True, "temperature": 0.85, "top_p": 0.9})
    else:
        gen_kwargs.update({"do_sample": False})
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return clean(generated)

class TextInput(BaseModel):
    text: str

class AnalyzePostRequest(BaseModel):
    text: str
    media_flags: Optional[List[str]] = None

class SummarizeThreadRequest(BaseModel):
    texts: List[str]
    image_counts: Optional[List[int]] = None
    image_alts: Optional[List[str]] = None

@app.get("/")
def root():
    return {"status": "running", "model": MODELNAME, "device": "GPU" if HASCUDA else "CPU"}

@app.post("/sentiment")
async def analyze_sentiment(data: TextInput):
    try:
        if not sentiment_pipe:
            return {"sentiment": {"label": "NEUTRAL", "confidence": 0.5}}
        
        result = sentiment_pipe(data.text[:512])[0]
        return {"sentiment": {"label": result["label"], "confidence": result["score"]}}
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"sentiment": {"label": "NEUTRAL", "confidence": 0.5}}

@app.post("/analyze")
async def analyze_content(data: TextInput):
    text = clean(data.text)
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "confidence": 0.0},
            "moderation": {"verdict": "safe", "confidence": 0.0, "labels": [], "reason": "Empty"},
            "raw_label": "",
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

Respond ONLY with JSON: {{"verdict":"safe/flagged/blocked","labels":[],"reason":"","confidence":0.0-1.0}}"""
        return generate(prompt, max_new=150, use_sampling=False)
    
    try:
        result = await asyncio.get_event_loop().run_in_executor(executor, mod_task)
        match = re.search(r'\{.*\}', result)
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
        },
        "raw_label": labels[0] if labels else "",
        "review_flag": verdict in ["flagged", "blocked"],
    }

@app.post("/rewrite")
async def rewrite_text(data: TextInput):
    if not data.text.strip():
        return {"rewritten_text": data.text}
    
    def task():
        prompt = f"""Rewrite this in a clear, natural way:

{data.text}"""
        return generate(prompt, max_new=200, use_sampling=True)
    
    try:
        draft = await asyncio.get_event_loop().run_in_executor(executor, task)
        draft = remove_emojis_and_symbols(draft.strip())
        draft = re.sub(r'#\w+', '', draft)
        draft = clean(draft)
        return {"rewritten_text": draft}
    except Exception as e:
        print(f"Rewrite error: {e}")
        return {"rewritten_text": data.text}

@app.post("/shorten")
async def shorten_text(data: TextInput):
    if not data.text.strip():
        return {"shortened_text": data.text}
    
    def task():
        prompt = f"""Make this shorter and concise:

{data.text}"""
        return generate(prompt, max_new=100, use_sampling=True)
    
    try:
        shortened = await asyncio.get_event_loop().run_in_executor(executor, task)
        shortened = remove_emojis_and_symbols(shortened.strip())
        shortened = re.sub(r'#\w+', '', shortened)
        shortened = clean(shortened)
        return {"shortened_text": shortened}
    except Exception as e:
        print(f"Shorten error: {e}")
        words = data.text.split()
        return {"shortened_text": " ".join(words[:50]) + ("..." if len(words) > 50 else "")}

@app.post("/summarize")
async def summarize_text(data: TextInput):
    if not data.text.strip():
        return {"summary": data.text}
    
    def task():
        prompt = f"""Summarize this clearly and conversationally:

{data.text}"""
        return generate(prompt, max_new=150, use_sampling=True)
    
    try:
        summary = await asyncio.get_event_loop().run_in_executor(executor, task)
        summary = clean(summary)
        return {"summary": summary}
    except Exception as e:
        print(f"Summarize error: {e}")
        words = data.text.split()
        return {"summary": " ".join(words[:100]) + ("..." if len(words) > 100 else "")}

@app.post("/api/analyze-post")
async def analyze_post(req: AnalyzePostRequest):
    return await analyze_content(TextInput(text=req.text))

@app.post("/api/summarize-thread")
async def summarize_thread(req: SummarizeThreadRequest):
    texts = [clean(t) for t in req.texts if t]
    if not texts:
        return {"summary": ""}
    
    context = " ".join(texts)
    
    if req.image_counts:
        img_count = sum(max(0, c) for c in req.image_counts)
        if img_count > 0:
            context = f"{img_count} {'images' if img_count != 1 else 'image'}. {context}"
    
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
        summary = re.sub(r'summary|the thread?', '', summary, flags=re.I).strip()
        
        if len(summary) > max_chars:
            sentences = re.split(r'[.!?]+', summary)
            result = []
            length = 0
            for sent in sentences:
                if length + len(sent) <= max_chars:
                    result.append(sent)
                    length += len(sent) + 1
                else:
                    break
            summary = '. '.join(result) if result else sentences[0][:max_chars]
        
        return {"summary": summary or "Discussion"}
    except:
        return {"summary": context[:100]}

@app.post("/api/condense-to-post")
async def condense(data: Dict[str, Any]):
    text = clean(data.get("text", ""))
    if not text:
        return {"draft": ""}
    
    word_count = len(text.split())
    target_ratio = 0.4 if word_count > 100 else 0.6
    target_words = max(int(word_count * target_ratio), 20)
    
    def task():
        prompt = f"""You are an expert writer who rewrites text to make it shorter, punchier, and more readable.

INSTRUCTIONS:
- Reduce length to roughly {target_words} words.
- Keep all essential meaning and logical flow.
- Ensure the final output fits entirely within {MAXPOSTCHARS} characters including spaces and punctuation.
- DO NOT exceed that character limit.
- Remove fluff, repetition, or weak phrasing.
- Write naturally like a human (not bullet points unless original is a list).
- Output only the rewritten version, ending on a complete sentence.
- No emojis, no hashtags.

TEXT:
{text}

COMPRESSED VERSION ({MAXPOSTCHARS} chars):"""
        return generate(prompt, max_new=512, use_sampling=True, char_limit=MAXPOSTCHARS)
    
    try:
        draft = await asyncio.get_event_loop().run_in_executor(executor, task)
        draft = remove_emojis_and_symbols(draft.strip())
        draft = re.sub(r'#\w+', '', draft)
        draft = clean(draft)
        
        if len(draft) > MAXPOSTCHARS:
            draft = trim_to_sentence_boundary(draft, MAXPOSTCHARS)
        
        return {"draft": draft}
    except Exception as e:
        print(f"Condense error: {e}")
        return {"draft": trim_to_sentence_boundary(text, MAXPOSTCHARS)}

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
- Ensure the final output fits entirely within {MAXPOSTCHARS} characters including spaces and punctuation.
- DO NOT exceed that character limit.
- No emojis, no hashtags.
- Output only the rewritten text and end on a complete sentence.

ORIGINAL:
{prompt_text}

REWRITTEN ({MAXPOSTCHARS} chars):"""
        return generate(prompt, max_new=512, use_sampling=True, char_limit=MAXPOSTCHARS)
    
    try:
        draft = await asyncio.get_event_loop().run_in_executor(executor, task)
        draft = remove_emojis_and_symbols(draft)
        draft = re.sub(r'#\w+', '', draft)
        draft = clean(draft)
        
        if len(draft) > MAXPOSTCHARS:
            draft = trim_to_sentence_boundary(draft, MAXPOSTCHARS)
        
        return {"draft": draft}
    except Exception as e:
        print(f"Draft error: {e}")
        return {"draft": trim_to_sentence_boundary(prompt_text, MAXPOSTCHARS)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)
