// src/services/aiService.ts
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function postJSON<T>(path: string, body: any): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${path} failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export interface AnalyzePostResponse {
  sentiment: { label: string; confidence: number };
  moderation: {
    verdict: "safe" | "flagged" | "blocked";
    confidence: number;
    labels: string[];
    reason: string;
    raw_label: string;
  };
  review_flag: boolean;
}

export default {
  analyzePost(text: string, media_flags?: string[]) {
    return postJSON<AnalyzePostResponse>("/api/analyze-post", { text, media_flags });
  },

  moderate(text: string, media_flags?: string[]) {
    return postJSON<{
      verdict: "safe" | "flagged" | "blocked";
      confidence: number;
      labels: string[];
      reason: string;
      raw_label: string;
      review_flag: boolean;
    }>("/api/moderate", { text, media_flags });
  },

  condenseToPost(text: string) {
    return postJSON<{ draft: string }>("/api/condense-to-post", { text });
  },

  summarizeThread(payload: { texts: string[]; image_counts?: number[]; image_alts?: string[] }) {
    return postJSON<{ summary: string }>("/api/summarize-thread", payload);
  },

  draftPost(prompt: string) {
    return postJSON<{ draft: string; error?: string }>("/api/draft-post", { prompt });
  },
};
