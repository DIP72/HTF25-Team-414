# backend/main.py
import os
import re
import json
import requests
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import pipeline

# ------------- Config -------------
HF_TOKEN = os.getenv("HF_TOKEN", "")  # optional; only needed if you later use HF hosted chat
HF_API_URL = "https://api-inference.huggingface.co/models"

# Local small chat model fallback (CPU friendly, ~0.5–0.8 GB)
LOCAL_GEN = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Retrieval
DUCKDUCKGO_SEARCH = "https://duckduckgo.com/html/?q="  # scrape-friendly
MAX_WEB_SNIPPETS = 5

# ------------- App -------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🤖 Loading models...")

device = 0 if torch.cuda.is_available() else -1
device_name = "GPU" if device == 0 else "CPU"
print(f"📊 Using device: {device_name}")

# Moderation / sentiment / summarization (local)
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=device)
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Local generator
try:
    local_gen = pipeline("text-generation", model=LOCAL_GEN, device=device)
    local_tok = local_gen.tokenizer
    print(f"✅ Local fallback model ready: {LOCAL_GEN}")
except Exception as e:
    local_gen, local_tok = None, None
    print(f"⚠️ Local fallback unavailable: {e}")

# ------------- Helpers -------------

BANNED = re.compile(r"\b(stupid|idiot|loser|kill yourself|hate you)\b", re.I)
FORBID_META = re.compile(r"\b(Dear user|I can|I will|As an AI|In this article|Here'?s what|allow me)\b", re.I)

def is_clean(text: str) -> bool:
    if len(text.split()) < 60:  # 60+ words ~ proper paragraph
        return False
    if BANNED.search(text) or FORBID_META.search(text):
        return False
    return True

def ddg_search_snippets(query: str, max_snippets: int = MAX_WEB_SNIPPETS) -> List[str]:
    try:
        import urllib.parse
        # Add recency bias + trusted domains
        q = urllib.parse.quote_plus(
            f"{query} 2023 2024 2025 site:reuters.com OR site:bloomberg.com OR site:ft.com OR site:apnews.com"
        )
        url = DUCKDUCKGO_SEARCH + q
        html = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
        items = re.findall(
            r'<a.*?class="result__a".*?>(.*?)</a>|<a.*?class="result__snippet".*?>(.*?)</a>',
            html,
            flags=re.I | re.S,
        )
        cleaned: List[str] = []
        for a, b in items:
            s = a or b
            s = re.sub("<.*?>", " ", s)
            s = re.sub(r"&[a-z]+;", " ", s)
            s = " ".join(s.split())
            if len(s.split()) >= 5:
                cleaned.append(s)
            if len(cleaned) >= max_snippets:
                break
        return cleaned
    except Exception:
        return []

def build_system(kind: str) -> str:
    base = (
        "You write concise, factual social posts.\n"
        "- One cohesive paragraph, 6–9 sentences.\n"
        "- No greetings, no meta narration, no lists, no headings.\n"
        "- Neutral, analytic tone. No questions. No imperatives.\n"
        "- Use details from snippets when present; never invent dates or figures."
    )
    if kind == "reply":
        base += "\n- End with one concrete takeaway."
    return base

def build_user(topic: str, snippets: List[str]) -> str:
    ctx = "\n".join(f"- {s}" for s in snippets) if snippets else "- (no snippets available)"
    return (
        f"Topic: {topic}\n"
        f"Context snippets (paraphrase, do not quote, do not invent numbers beyond these):\n{ctx}\n\n"
        "Write one paragraph (6–9 sentences). If figures are missing, stay qualitative."
    )

def chat_template(system: str, user: str) -> str:
    # TinyLlama / Alpaca style
    return f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"

def generate_llm(prompt: str, max_new: int = 260) -> str:
    if local_gen is None:
        return "Local generator unavailable."
    # Encode with attention_mask to avoid warnings and stabilize CPU decoding
    enc = local_tok(prompt, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(local_gen.model.device)
    attn = enc["attention_mask"].to(local_gen.model.device)
    out_ids = local_gen.model.generate(
        input_ids,
        attention_mask=attn,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        max_new_tokens=max_new,
        eos_token_id=local_tok.eos_token_id,
        pad_token_id=local_tok.eos_token_id,
    )
    text = local_tok.decode(out_ids[0], skip_special_tokens=True)
    out = text.split("### Assistant:")[-1].strip()
    out = " ".join(out.split())
    # keep ~170 words
    words = out.split()
    if len(words) > 170:
        out = " ".join(words[:170]) + "..."
    if not out.endswith("."):
        out += "."
    return out

def enforce_number_if_present(draft: str, snippets: List[str]) -> str:
    has_num = any(re.search(r"\d", s) for s in snippets)
    if has_num and not re.search(r"\d", draft):
        # append one short numeric clause paraphrased from first numeric snippet
        num_snip = next((s for s in snippets if re.search(r"\d", s)), "")
        clause = re.sub(r"^-\s*", "", num_snip).split(".")[0]
        clause = " ".join(clause.split())[:140]
        if clause and clause not in draft:
            draft = draft.rstrip(".") + f". {clause}."
    return draft

def generate_paragraph(topic: str, kind: str, snippets: List[str]) -> str:
    system = build_system(kind)
    user = build_user(topic, snippets)
    prompt = chat_template(system, user)

    draft = generate_llm(prompt, max_new=280)
    # Quality gate: meta/length; one controlled regen
    if not is_clean(draft):
        draft = generate_llm(prompt, max_new=320)
    draft = re.sub(r"^(Dear .*?,\s*)", "", draft)
    draft = re.sub(FORBID_META, "", draft)
    draft = enforce_number_if_present(draft, snippets)
    return draft.strip()

# ------------- API -------------

@app.get("/")
def root():
    return {"status": "running", "device": device_name, "local_fallback": LOCAL_GEN}

@app.post("/api/moderate")
def moderate(data: Dict[str, Any]):
    text = (data.get("text") or "").strip()
    if len(text) < 3:
        return {"verdict": "safe", "confidence": 1.0, "labels": [], "reason": "Too short"}
    r = toxicity_classifier(text[:512])[0]
    score = r["score"]
    labels = []
    if "toxic" in r["label"].lower() and score > 0.5:
        labels.append("Toxic")
    if score > 0.7:
        labels.append("Offensive")
    verdict = "blocked" if score > 0.8 else "flagged" if score > 0.5 else "safe"
    reason = "High toxicity" if score > 0.8 else "Potentially inappropriate" if score > 0.5 else "Safe"
    return {"verdict": verdict, "confidence": score, "labels": labels, "reason": reason, "raw_label": r["label"]}

@app.post("/api/summarize")
def summarize_thread(data: Dict[str, Any]):
    texts: List[str] = data.get("texts") or []
    if not texts:
        return {"summary": "No content"}
    merged = " ".join(texts)[:3000]
    if len(merged.split()) < 50:
        return {"summary": "Thread too short", "original_length": len(merged.split())}
    s = summarizer(merged, max_length=140, min_length=40, do_sample=False)[0]["summary_text"]
    return {"summary": s, "original_length": len(merged.split()), "summary_length": len(s.split())}

@app.post("/api/draft-post")
def draft_post(data: Dict[str, Any]):
    topic = (data.get("prompt") or "").strip()
    use_web = bool(data.get("web", True))
    if len(topic) < 3:
        return {"draft": "", "error": "Prompt too short"}
    snippets = ddg_search_snippets(topic, MAX_WEB_SNIPPETS) if use_web else []
    draft = generate_paragraph(topic, "post", snippets)
    return {"draft": draft, "used_web": bool(snippets), "snippets_preview": snippets[:2]}

@app.post("/api/draft-reply")
def draft_reply(data: Dict[str, Any]):
    context = (data.get("prompt") or "").strip()
    use_web = bool(data.get("web", True))
    if len(context) < 3:
        return {"draft": "", "error": "Context too short"}
    snippets = ddg_search_snippets(context, 4) if use_web else []
    draft = generate_paragraph(context, "reply", snippets)
    return {"draft": draft, "used_web": bool(snippets), "snippets_preview": snippets[:2]}

# Optional alias if your frontend calls analyzePost
@app.post("/api/analyze-post")
def analyze_post(data: Dict[str, Any]):
    text = (data.get("text") or "").strip()
    r = toxicity_classifier(text[:512])[0]
    s = sentiment_analyzer(text[:512])[0]
    score = r["score"]
    labels = []
    if "toxic" in r["label"].lower() and score > 0.5:
        labels.append("Toxic")
    if score > 0.7:
        labels.append("Offensive")
    verdict = "blocked" if score > 0.8 else "flagged" if score > 0.5 else "safe"
    reason = "High toxicity" if score > 0.8 else "Potentially inappropriate" if score > 0.5 else "Safe"
    return {
        "moderation": {"verdict": verdict, "confidence": score, "labels": labels, "raw_label": r["label"], "reason": reason},
        "sentiment": {"label": s["label"], "confidence": s["score"]},
        "flagged": score > 0.5,
        "safe_to_post": score < 0.8
    }

@app.post("/api/analyzePost")
def analyze_post_alias(data: Dict[str, Any]):
    return analyze_post(data)

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
