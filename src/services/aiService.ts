const getApiUrl = (): string => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  return 'http://localhost:8000';
};

const API_URL = getApiUrl();

interface AnalysisResult {
  sentiment: {
    label: string;
    confidence: number;
  };
  moderation?: {
    verdict: string;
    confidence: number;
    labels?: string[];
    reason?: string;
  };
  labels?: string[];
  message?: string;
  review_flag?: boolean;
}

class AIService {
  async analyzePost(content: string, mediaFlags?: string[]): Promise<AnalysisResult> {
    try {
      const response = await fetch(`${API_URL}/api/analyze-post`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: content, media_flags: mediaFlags || [] })
      });

      if (!response.ok) throw new Error('Analysis failed');
      
      const data = await response.json();
      return {
        sentiment: data.sentiment,
        moderation: data.moderation,
        review_flag: data.review_flag
      };
    } catch (error) {
      console.error('Analysis error:', error);
      return {
        sentiment: { label: 'NEUTRAL', confidence: 0.5 },
        moderation: { verdict: 'safe', confidence: 0.9 },
        labels: [],
        message: 'Analysis unavailable'
      };
    }
  }

  async getSentimentOnly(content: string): Promise<{ sentiment: { label: string; confidence: number } }> {
    try {
      const response = await fetch(`${API_URL}/api/sentiment-only`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: content })
      });

      if (!response.ok) throw new Error('Sentiment analysis failed');
      return await response.json();
    } catch (error) {
      console.error('Sentiment error:', error);
      return { sentiment: { label: 'NEUTRAL', confidence: 0.5 } };
    }
  }

  async draftPost(prompt: string): Promise<{ draft: string }> {
    try {
      const response = await fetch(`${API_URL}/api/draft-post`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      if (!response.ok) throw new Error('Draft generation failed');
      return await response.json();
    } catch (error) {
      console.error('Draft error:', error);
      throw error;
    }
  }

  async condenseToPost(text: string): Promise<{ draft: string }> {
    try {
      const response = await fetch(`${API_URL}/api/condense-to-post`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) throw new Error('Condense failed');
      return await response.json();
    } catch (error) {
      console.error('Condense error:', error);
      throw error;
    }
  }

  async summarizeThread(texts: string[], imageCounts?: number[], imageAlts?: string[]): Promise<{ summary: string }> {
    try {
      const response = await fetch(`${API_URL}/api/summarize-thread`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts, image_counts: imageCounts, image_alts: imageAlts })
      });

      if (!response.ok) throw new Error('Thread summarization failed');
      return await response.json();
    } catch (error) {
      console.error('Thread summarization error:', error);
      return { summary: texts[0]?.substring(0, 100) + '...' || 'Discussion' };
    }
  }
}

export default new AIService();
