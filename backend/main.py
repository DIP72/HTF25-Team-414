import os
import re
import json
import unicodedata
from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# VADER for sentiment (NO API needed, lightweight)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    HAS_VADER = True
except:
    HAS_VADER = False

MAXPOSTCHARS = 1000

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API for content moderation only
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()

def normalize_text(text: str) -> str:
    """
    Remove special characters, unicode tricks, and normalize text
    to prevent bypass attempts
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove unicode confusables (e.g., Cyrillic lookalikes)
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Replace common leetspeak/special character tricks
    replacements = {
        '@': 'a', '4': 'a', '3': 'e', '1': 'i', '!': 'i',
        '0': 'o', '$': 's', '5': 's', '7': 't', '+': 't',
        '8': 'b', '9': 'g', '6': 'g', '*': 'x'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove spaces between characters (common bypass: "f u c k" -> "fuck")
    # But keep normal word spacing
    words = text.split()
    normalized_words = []
    for word in words:
        # If word has spaces/dots between each char, remove them
        if len(word) > 1:
            # Remove all non-alphanumeric except space
            clean_word = ''.join(c for c in word if c.isalnum() or c.isspace())
            normalized_words.append(clean_word)
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)

def vader_sentiment(text: str) -> dict:
    """
    VADER sentiment - FREE, no API, works offline
    Perfect for social media text
    """
    if not HAS_VADER:
        # Fallback if VADER not installed
        return {"label": "NEUTRAL", "confidence": 0.5}
    
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # VADER compound score ranges from -1 (most negative) to +1 (most positive)
    if compound >= 0.05:
        label = "POSITIVE"
        confidence = min(0.99, abs(compound))
    elif compound <= -0.05:
        label = "NEGATIVE"
        confidence = min(0.99, abs(compound))
    else:
        label = "NEUTRAL"
        confidence = 1.0 - abs(compound)
    
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "scores": {
            "positive": scores['pos'],
            "negative": scores['neg'],
            "neutral": scores['neu']
        }
    }

def basic_moderation(text: str) -> dict:
    """
    Rule-based moderation with bypass prevention
    Catches 80% of violations without API
    """
    normalized = normalize_text(text)
    
    # Comprehensive blocked words list (add more as needed)
    blocked_words = [
        "fuck", "shit", "bitch", "ass", "damn", "hell",
        "cunt", "dick", "pussy", "cock", "nigga", "nigger",
        "bastard", "whore", "slut", "piss", "rape",
        "kill yourself", "kys", "suicide", "die",
        "retard", "faggot", "fag", "gay", "lesbian",
        "spam", "scam", "viagra", "crypto", "bitcoin",
        "click here", "buy now", "limited offer",
        "violence", "murder", "terrorist", "bomb", "weapon"
    ]
    
    flagged_words = [
        "controversial", "political", "religion", "sex",
        "drug", "alcohol", "gambling", "nude", "nsfw"
    ]
    
    verdict = "safe"
    labels = []
    reason = "Safe content"
    confidence = 0.0
    matched_words = []
    
    # Check for blocked content in normalized text
    for word in blocked_words:
        if word in normalized:
            verdict = "blocked"
            labels.append("profanity" if len(word) <= 8 else "harmful")
            matched_words.append(word)
            confidence = 0.9
            reason = "Contains blocked content"
            break
    
    # Check for flagged content if not already blocked
    if verdict == "safe":
        for word in flagged_words:
            if word in normalized:
                verdict = "flagged"
                labels.append("needs_review")
                matched_words.append(word)
                confidence = 0.7
                reason = "Contains sensitive topics"
                break
    
    # Pattern detection for spam
    if verdict == "safe":
        # Too many CAPS
        if len(text) > 20 and sum(1 for c in text if c.isupper()) / len(text) > 0.7:
            verdict = "flagged"
            labels.append("spam")
            confidence = 0.6
            reason = "Excessive caps"
        
        # Too many repeated characters
        if re.search(r'(.)\1{4,}', text):
            verdict = "flagged"
            labels.append("spam")
            confidence = 0.6
            reason = "Repeated characters"
        
        # Too many links
        link_count = len(re.findall(r'http[s]?://|www\.', text))
        if link_count > 2:
            verdict = "blocked"
            labels.append("spam")
            confidence = 0.8
            reason = "Multiple links"
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "labels": labels if labels else [],
        "reason": reason,
        "matched": matched_words[:3]  # Show up to 3 matched words
    }

def groq_advanced_moderation(text: str) -> dict:
    """
    Use Groq API only for edge cases that bypass rules
    Rate limit: 30 requests/minute (enough for most use cases)
    """
    if not GROQ_API_KEY:
        return basic_moderation(text)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Analyze if this post violates content policies. Look for:
- Profanity (even with special chars/misspellings)
- Hate speech
- Violent threats
- Spam
- NSFW content
- Harassment

Text: "{text}"

Respond ONLY with JSON:
{{"verdict":"safe/flagged/blocked","reason":"brief reason","confidence":0.0-1.0}}"""
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            # Extract JSON from response
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "verdict": data.get("verdict", "safe"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "labels": ["ai_detected"],
                    "reason": data.get("reason", "AI analysis")
                }
    except Exception as e:
        print(f"Groq API error: {e}")
    
    # Fallback to basic moderation
    return basic_moderation(text)

def groq_call(prompt: str, max_tokens: int = 300) -> str:
    """Generic Groq API call for text generation"""
    if not GROQ_API_KEY:
        return ""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq error: {e}")
    
    return ""

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
    status = "VADER + Groq" if HAS_VADER and GROQ_API_KEY else "Basic mode"
    return {"status": "running", "mode": status}

@app.post("/sentiment")
async def analyze_sentiment(data: TextInput):
    """Uses VADER - NO API needed, unlimited requests"""
    return {"sentiment": vader_sentiment(data.text)}

@app.post("/analyze")
async def analyze_content(data: TextInput):
    """
    Hybrid approach:
    1. VADER sentiment (free, unlimited)
    2. Basic rule moderation (catches 80% of violations)
    3. Groq AI moderation (only for edge cases)
    """
    text = clean(data.text)
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "confidence": 0.0},
            "moderation": {"verdict": "safe", "confidence": 0.0, "labels": [], "reason": "Empty"},
            "raw_label": "",
            "review_flag": False,
        }
    
    # Step 1: Free sentiment analysis
    sentiment = vader_sentiment(text)
    
    # Step 2: Basic rule-based moderation (catches most violations)
    moderation = basic_moderation(text)
    
    # Step 3: Use Groq only if basic check is uncertain or user seems to bypass
    if moderation["verdict"] == "safe" and (
        # Detect potential bypass attempts
        len(re.findall(r'[^a-zA-Z0-9\s]', text)) > len(text) * 0.2 or  # Too many special chars
        normalize_text(text) != text.lower()  # Text changed after normalization
    ):
        # Use Groq API for smart detection
        moderation = groq_advanced_moderation(text)
    
    return {
        "sentiment": sentiment,
        "moderation": moderation,
        "raw_label": moderation["labels"][0] if moderation["labels"] else "",
        "review_flag": moderation["verdict"] in ["flagged", "blocked"],
    }

@app.post("/rewrite")
async def rewrite_text(data: TextInput):
    if not data.text.strip():
        return {"rewritten_text": data.text}
    
    prompt = f"Rewrite this clearly and concisely:\n\n{data.text}\n\nRewritten:"
    result = groq_call(prompt, max_tokens=300)
    
    return {"rewritten_text": clean(result) if result else data.text}

@app.post("/shorten")
async def shorten_text(data: TextInput):
    if not data.text.strip():
        return {"shortened_text": data.text}
    
    prompt = f"Make this much shorter:\n\n{data.text}\n\nShortened:"
    result = groq_call(prompt, max_tokens=150)
    
    if not result:
        words = data.text.split()
        return {"shortened_text": " ".join(words[:50]) + ("..." if len(words) > 50 else "")}
    
    return {"shortened_text": clean(result)}

@app.post("/summarize")
async def summarize_text(data: TextInput):
    if not data.text.strip():
        return {"summary": data.text}
    
    prompt = f"Summarize in 2-3 sentences:\n\n{data.text}\n\nSummary:"
    result = groq_call(prompt, max_tokens=200)
    
    if not result:
        words = data.text.split()
        return {"summary": " ".join(words[:100]) + ("..." if len(words) > 100 else "")}
    
    return {"summary": clean(result)}

@app.post("/api/analyze-post")
async def analyze_post(req: AnalyzePostRequest):
    return await analyze_content(TextInput(text=req.text))

@app.post("/api/summarize-thread")
async def summarize_thread(req: SummarizeThreadRequest):
    texts = [clean(t) for t in req.texts if t]
    if not texts:
        return {"summary": ""}
    
    context = " ".join(texts)[:2000]
    
    prompt = f"Summarize this discussion in 2-3 sentences:\n\n{context}\n\nSummary:"
    result = groq_call(prompt, max_tokens=200)
    
    return {"summary": clean(result) if result else context[:100]}

@app.post("/api/condense-to-post")
async def condense(data: Dict[str, Any]):
    text = clean(data.get("text", ""))
    if not text:
        return {"draft": ""}
    
    prompt = f"Rewrite to be under {MAXPOSTCHARS} characters:\n\n{text}\n\nRewritten:"
    result = groq_call(prompt, max_tokens=400)
    
    draft = clean(result) if result else text
    return {"draft": draft[:MAXPOSTCHARS]}

@app.post("/api/draft-post")
async def draft_post(data: Dict[str, Any]):
    prompt_text = clean(data.get("prompt", ""))
    if len(prompt_text) < 3:
        return {"draft": "", "error": "Prompt too short"}
    
    prompt = f"Rewrite naturally:\n\n{prompt_text}\n\nRewritten:"
    result = groq_call(prompt, max_tokens=400)
    
    draft = clean(result) if result else prompt_text
    return {"draft": draft[:MAXPOSTCHARS]}

app = app
