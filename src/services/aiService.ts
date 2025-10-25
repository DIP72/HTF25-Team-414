// src/services/aiService.ts

const API_BASE = "http://localhost:8000/api";

export interface ModerationResult {
  verdict: "safe" | "flagged" | "blocked";
  confidence: number;
  labels: string[];
  reason: string;  // ✅ Added this
  raw_label: string;
}

export interface DraftResult {
  draft: string;
  prompt?: string;
  context?: string;
  length: number;
}

export interface SummaryResult {
  summary: string;
  original_length: number;
  summary_length?: number;
}

export interface AnalysisResult {
  moderation: {
    verdict: string;
    confidence: number;
    labels: string[];
    raw_label: string;
    reason: string;  // ✅ Added this
  };
  sentiment: {
    label: string;
    confidence: number;
  };
  flagged: boolean;
  safe_to_post: boolean;
}

export const aiService = {
  async moderateContent(text: string): Promise<ModerationResult> {
    const response = await fetch(`${API_BASE}/moderate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    return response.json();
  },

  async draftPost(prompt: string, maxLength: number = 100): Promise<DraftResult> {
    const response = await fetch(`${API_BASE}/draft-post`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, max_length: maxLength }),
    });
    return response.json();
  },

  async draftReply(context: string, maxLength: number = 80): Promise<DraftResult> {
    const response = await fetch(`${API_BASE}/draft-reply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: context, max_length: maxLength }),
    });
    return response.json();
  },

  async summarizeThread(texts: string[]): Promise<SummaryResult> {
    const response = await fetch(`${API_BASE}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts, max_length: 130, min_length: 30 }),
    });
    return response.json();
  },

  async analyzePost(text: string): Promise<AnalysisResult> {
    const response = await fetch(`${API_BASE}/analyze-post`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    return response.json();
  },
};
