const API_BASE = "http://localhost:8000";

interface AnalysisResult {
  sentiment: {
    label: string;
    confidence: number;
  };
  moderation: {
    verdict: "safe" | "flagged" | "blocked";
    confidence: number;
    labels: string[];
    reason: string;
    raw_label: string;
  };
  review_flag: boolean;
}

interface SentimentOnlyResult {
  sentiment: {
    label: string;
    confidence: number;
  };
}

interface SummarizeThreadRequest {
  texts: string[];
  image_counts?: number[];
  image_alts?: string[];
}

interface SummarizeThreadResponse {
  summary: string;
}

interface DraftPostResponse {
  draft: string;
  error?: string;
}

interface CondensePostResponse {
  draft: string;
}

class AIService {
  private async postJSON<T>(endpoint: string, data: any): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return response.json();
  }

  async analyzePost(text: string, media_flags?: string[]): Promise<AnalysisResult> {
    return this.postJSON<AnalysisResult>("/api/analyze-post", {
      text,
      media_flags: media_flags || [],
    });
  }

  async getSentimentOnly(text: string): Promise<SentimentOnlyResult> {
    return this.postJSON<SentimentOnlyResult>("/api/sentiment-only", { text });
  }

  async moderatePost(text: string, media_flags?: string[]) {
    return this.postJSON("/api/moderate", {
      text,
      media_flags: media_flags || [],
    });
  }

  async summarizeThread(request: SummarizeThreadRequest): Promise<SummarizeThreadResponse> {
    return this.postJSON<SummarizeThreadResponse>("/api/summarize-thread", {
      texts: request.texts,
      image_counts: request.image_counts || [],
      image_alts: request.image_alts || [],
    });
  }

  async draftPost(prompt: string): Promise<DraftPostResponse> {
    return this.postJSON<DraftPostResponse>("/api/draft-post", {
      prompt,
    });
  }

  async condenseToPost(text: string): Promise<CondensePostResponse> {
    return this.postJSON<CondensePostResponse>("/api/condense-to-post", {
      text,
    });
  }

  async healthCheck() {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
  }
}

const aiService = new AIService();
export default aiService;
