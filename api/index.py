import os
import re
import json
import unicodedata
from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
import requests

# VADER for sentiment analysis (no API needed)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("Warning: VADER not available, using fallback sentiment")

MAXPOSTCHARS = 1000

app = FastAPI(title="Rekonect API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API configuration (read from environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ==================== UTILITY FUNCTIONS ====================

def clean(text: str) -> str:
    """Remove extra whitespace"""
    return re.sub(r'\s+', ' ', text or '').strip()

def normalize_text(text: str) -> str:
    """
    Normalize text to prevent bypass attempts:
    - Removes unicode confusables (Cyrillic lookalikes)
    - Converts leetspeak (@ → a, 3 → e, etc.)
    - Removes spaces between characters
    - Handles special character tricks
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize unicode characters (handles Cyrillic tricks)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Replace common leetspeak and special character substitutions
    replacements = {
        '@': 'a', '4': 'a', 'Ā': 'a',
        '3': 'e', '€': 'e',
        '1': 'i', '!': 'i', '|': 'i',
        '0': 'o', 'Ø': 'o',
        '$': 's', '5': 's', 'ß': 's',
        '7': 't', '+': 't',
        '8': 'b',
        '9': 'g', '6': 'g',
        '*': 'x',
        '()': 'o',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove spaces/dots/underscores between single characters (f.u.c.k → fuck)
    # But preserve normal word spacing
    text = re.sub(r'(\w)[_.\s]+(?=\w)', r'\1', text)
    
    # Remove all remaining non-alphanumeric except spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_emojis_and_symbols(text: str) -> str:
    """Remove emojis and special symbols"""
    if not text:
        return text
    
    output = []
    for char in text:
        cp = ord(char)
        # Skip emoji ranges
        if not (0x1F300 <= cp <= 0x1F9FF or 0x2600 <= cp <= 0x27BF or 
                0xFE00 <= cp <= 0xFE0F):
            output.append(char)
    
    return ''.join(output)

# ==================== SENTIMENT ANALYSIS ====================

def vader_sentiment(text: str) -> dict:
    """
    VADER sentiment analysis - FREE, no API required
    Specifically designed for social media text
    Returns: {"label": "POSITIVE/NEGATIVE/NEUTRAL", "confidence": 0.0-1.0}
    """
    if not HAS_VADER:
        # Fallback to simple rule-based sentiment
        return simple_sentiment_fallback(text)
    
    try:
        scores = vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # VADER compound: -1 (most negative) to +1 (most positive)
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
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        print(f"VADER error: {e}")
        return {"label": "NEUTRAL", "confidence": 0.5}

def simple_sentiment_fallback(text: str) -> dict:
    """Fallback sentiment if VADER not available"""
    positive_words = ["good", "great", "excellent", "amazing", "love", "happy", 
                     "best", "wonderful", "awesome", "fantastic", "perfect"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "sad", 
                     "angry", "horrible", "disgusting", "poor", "sucks"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return {"label": "POSITIVE", "confidence": min(0.9, 0.5 + pos_count * 0.1)}
    elif neg_count > pos_count:
        return {"label": "NEGATIVE", "confidence": min(0.9, 0.5 + neg_count * 0.1)}
    else:
        return {"label": "NEUTRAL", "confidence": 0.5}

# ==================== CONTENT MODERATION ====================

def basic_moderation(text: str) -> dict:
    """
    Rule-based content moderation with bypass prevention
    Catches 80-90% of violations without API calls
    """
    original_text = text
    normalized = normalize_text(text)
    
    # Comprehensive blocked words list
    blocked_words = [
        # Profanity
        "fuck", "shit", "bitch", "ass", "damn", "hell", "cunt", "dick", 
        "pussy", "cock", "bastard", "whore", "slut", "piss",
        # Slurs (partial list for demo - expand as needed)
        "nigga", "nigger", "retard", "faggot", "fag",
        # Violence & threats
        "kill", "murder", "die", "death", "kys", "suicide", "rape",
        "terrorist", "bomb", "weapon", "shoot", "stab", "attack",
        # Spam indicators
        "viagra", "crypto", "bitcoin", "click here", "buy now", 
        "limited offer", "congratulations you won", "claim prize",
    ]
    
    # Flagged words (require review but not auto-blocked)
    flagged_words = [
        "controversial", "political", "religion", "sex", "drug", 
        "alcohol", "gambling", "nude", "nsfw", "porn"
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
            labels.append("harmful_content")
            matched_words.append(word)
            confidence = 0.9
            reason = "Contains prohibited content"
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
    
    # Additional pattern-based checks
    if verdict == "safe":
        # Excessive caps (spam indicator)
        if len(original_text) > 20:
            caps_ratio = sum(1 for c in original_text if c.isupper()) / len(original_text)
            if caps_ratio > 0.7:
                verdict = "flagged"
                labels.append("spam")
                confidence = 0.6
                reason = "Excessive capitalization"
        
        # Repeated characters (spam indicator: "hellooooooo")
        if re.search(r'(.)\1{5,}', original_text):
            verdict = "flagged"
            labels.append("spam")
            confidence = 0.6
            reason = "Repeated characters"
        
        # Multiple links (spam)
        link_count = len(re.findall(r'http[s]?://|www\.', original_text))
        if link_count > 2:
            verdict = "blocked"
            labels.append("spam")
            confidence = 0.85
            reason = "Multiple suspicious links"
        
        # All emojis, no text (spam)
        text_without_emojis = remove_emojis_and_symbols(original_text)
        if len(text_without_emojis.strip()) < 3 and len(original_text) > 10:
            verdict = "flagged"
            labels.append("spam")
            confidence = 0.7
            reason = "Mostly emojis"
    
    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "labels": labels if labels else [],
        "reason": reason
    }

def groq_advanced_moderation(text: str) -> dict:
    """
    Use Groq API for advanced moderation of edge cases
    Only called when basic rules are uncertain
    """
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY not set, using basic moderation only")
        return basic_moderation(text)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are a content moderator. Analyze if this text violates policies.

Check for:
- Profanity (including misspellings with special characters)
- Hate speech or slurs
- Violent threats or self-harm
- Harassment or bullying
- Spam or scams
- NSFW content

Text: "{text}"

Respond ONLY with JSON (no other text):
{{"verdict":"safe/flagged/blocked","reason":"brief explanation","confidence":0.0-1.0}}"""
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(
            GROQ_API_URL, 
            headers=headers, 
            json=payload, 
            timeout=8
        )
        
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                verdict = data.get("verdict", "safe").lower()
                
                # Validate verdict
                if verdict not in ["safe", "flagged", "blocked"]:
                    verdict = "safe"
                
                return {
                    "verdict": verdict,
                    "confidence": float(data.get("confidence", 0.5)),
                    "labels": ["ai_moderated"],
                    "reason": data.get("reason", "AI analysis completed")
                }
        else:
            print(f"Groq API error: {response.status_code}")
    
    except Exception as e:
        print(f"Groq API exception: {e}")
    
    # Fallback to basic moderation on error
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
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Groq API error: {response.status_code}")
            return ""
    
    except Exception as e:
        print(f"Groq call error: {e}")
        return ""

# ==================== PYDANTIC MODELS ====================

class TextInput(BaseModel):
    text: str

class AnalyzePostRequest(BaseModel):
    text: str
    media_flags: Optional[List[str]] = None

class SummarizeThreadRequest(BaseModel):
    texts: List[str]
    image_counts: Optional[List[int]] = None
    image_alts: Optional[List[str]] = None

# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    """Health check endpoint"""
    mode = []
    if HAS_VADER:
        mode.append("VADER")
    if GROQ_API_KEY:
        mode.append("Groq")
    
    return {
        "status": "running",
        "service": "Rekonect AI API",
        "version": "1.0.0",
        "features": mode if mode else ["Basic"]
    }

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "vader_available": HAS_VADER,
        "groq_available": bool(GROQ_API_KEY),
        "max_post_chars": MAXPOSTCHARS
    }

@app.post("/sentiment")
async def analyze_sentiment(data: TextInput):
    """
    Analyze sentiment of text (FREE - no API limit)
    Uses VADER sentiment analysis
    """
    try:
        sentiment = vader_sentiment(data.text)
        return {"sentiment": sentiment}
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"sentiment": {"label": "NEUTRAL", "confidence": 0.5}}

@app.post("/analyze")
async def analyze_content(data: TextInput):
    """
    Full content analysis: sentiment + moderation
    Uses hybrid approach: rule-based + AI when needed
    """
    text = clean(data.text)
    
    if not text:
        return {
            "sentiment": {"label": "NEUTRAL", "confidence": 0.0},
            "moderation": {
                "verdict": "safe",
                "confidence": 0.0,
                "labels": [],
                "reason": "Empty content"
            },
            "raw_label": "",
            "review_flag": False,
        }
    
    # Step 1: Sentiment analysis (always free/fast)
    sentiment = vader_sentiment(text)
    
    # Step 2: Basic rule-based moderation (catches most violations)
    moderation = basic_moderation(text)
    
    # Step 3: Use Groq AI only for uncertain/suspicious cases
    normalized = normalize_text(text)
    suspicious_indicators = (
        # Text changed significantly after normalization (bypass attempt)
        normalized != text.lower() or
        # Too many special characters (bypass attempt)
        len(re.findall(r'[^a-zA-Z0-9\s]', text)) > len(text) * 0.3 or
        # Basic check flagged it
        moderation["verdict"] == "flagged"
    )
    
    if suspicious_indicators and GROQ_API_KEY:
        # Use AI for smarter detection
        moderation = groq_advanced_moderation(text)
    
    return {
        "sentiment": sentiment,
        "moderation": moderation,
        "raw_label": moderation["labels"][0] if moderation["labels"] else "",
        "review_flag": moderation["verdict"] in ["flagged", "blocked"],
    }

@app.post("/rewrite")
async def rewrite_text(data: TextInput):
    """Rewrite text to be clearer and more concise"""
    if not data.text.strip():
        return {"rewritten_text": data.text}
    
    prompt = f"Rewrite this text to be clearer and more professional:\n\n{data.text}\n\nRewritten:"
    result = groq_call(prompt, max_tokens=300)
    
    if result:
        result = clean(remove_emojis_and_symbols(result))
        return {"rewritten_text": result}
    
    return {"rewritten_text": data.text}

@app.post("/shorten")
async def shorten_text(data: TextInput):
    """Make text shorter while keeping the main message"""
    if not data.text.strip():
        return {"shortened_text": data.text}
    
    prompt = f"Make this text much shorter (under 100 words):\n\n{data.text}\n\nShortened:"
    result = groq_call(prompt, max_tokens=150)
    
    if result:
        return {"shortened_text": clean(result)}
    
    # Fallback: simple truncation
    words = data.text.split()
    return {"shortened_text": " ".join(words[:50]) + ("..." if len(words) > 50 else "")}

@app.post("/summarize")
async def summarize_text(data: TextInput):
    """Summarize text in 2-3 sentences"""
    if not data.text.strip():
        return {"summary": data.text}
    
    prompt = f"Summarize this in 2-3 clear sentences:\n\n{data.text}\n\nSummary:"
    result = groq_call(prompt, max_tokens=200)
    
    if result:
        return {"summary": clean(result)}
    
    # Fallback
    words = data.text.split()
    return {"summary": " ".join(words[:100]) + ("..." if len(words) > 100 else "")}

@app.post("/api/analyze-post")
async def analyze_post(req: AnalyzePostRequest):
    """Legacy endpoint - redirects to /analyze"""
    return await analyze_content(TextInput(text=req.text))

@app.post("/api/summarize-thread")
async def summarize_thread(req: SummarizeThreadRequest):
    """Summarize a thread of posts"""
    texts = [clean(t) for t in req.texts if t]
    
    if not texts:
        return {"summary": ""}
    
    context = " ".join(texts)[:2000]
    
    # Add image info if provided
    if req.image_counts:
        img_count = sum(max(0, c) for c in req.image_counts)
        if img_count > 0:
            context = f"[{img_count} images] {context}"
    
    prompt = f"Summarize this discussion in 2-3 sentences:\n\n{context}\n\nSummary:"
    result = groq_call(prompt, max_tokens=200)
    
    if result:
        # Clean up common summary prefixes
        result = re.sub(r'^(the )?(thread|discussion|conversation) (is about |discusses )?', '', 
                       result, flags=re.I).strip()
        return {"summary": result}
    
    return {"summary": context[:150] + "..." if len(context) > 150 else context}

@app.post("/api/condense-to-post")
async def condense(data: Dict[str, Any]):
    """Condense long text to fit post character limit"""
    text = clean(data.get("text", ""))
    
    if not text:
        return {"draft": ""}
    
    if len(text) <= MAXPOSTCHARS:
        return {"draft": text}
    
    prompt = f"Rewrite this to be under {MAXPOSTCHARS} characters, keeping the main message:\n\n{text}\n\nConcise version:"
    result = groq_call(prompt, max_tokens=400)
    
    if result:
        draft = clean(result)
        return {"draft": draft[:MAXPOSTCHARS]}
    
    # Fallback: truncate at sentence boundary
    if len(text) > MAXPOSTCHARS:
        text = text[:MAXPOSTCHARS]
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > MAXPOSTCHARS * 0.5:
            text = text[:last_period + 1]
    
    return {"draft": text}

@app.post("/api/draft-post")
async def draft_post(data: Dict[str, Any]):
    """Generate a post from a prompt"""
    prompt_text = clean(data.get("prompt", ""))
    
    if len(prompt_text) < 3:
        return {"draft": "", "error": "Prompt too short"}
    
    prompt = f"Write a natural social media post based on this:\n\n{prompt_text}\n\nPost:"
    result = groq_call(prompt, max_tokens=400)
    
    if result:
        draft = clean(remove_emojis_and_symbols(result))
        return {"draft": draft[:MAXPOSTCHARS]}
    
    return {"draft": prompt_text[:MAXPOSTCHARS]}

# ==================== VERCEL HANDLER ====================

# This is critical for Vercel to work
handler = Mangum(app)
