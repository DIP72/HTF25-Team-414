const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const aiService = {
  async analyzePost(text: string): Promise<{
    sentiment: { label: string; confidence: number };
    moderation: { verdict: string; labels?: string[]; reason?: string };
    review_flag: boolean;
  }> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`);
      }

      return await response.json();
    } catch (error: any) {
      console.error('Analysis error:', error);
      throw error;
    }
  },

  async getSentimentOnly(text: string): Promise<{
    sentiment: { label: string; confidence: number };
  }> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/sentiment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Sentiment analysis failed: ${response.status}`);
      }

      return await response.json();
    } catch (error: any) {
      console.error('Sentiment error:', error);
      return { sentiment: { label: 'NEUTRAL', confidence: 0.5 } };
    }
  },

  async summarizeThread(
    texts: string[],
    image_counts: number[]
  ): Promise<{ summary: string }> {
    try {
      console.log('Summarizing thread:', { texts, image_counts });

      const response = await fetch(`${API_BASE_URL}/api/summarize-thread`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts, image_counts }),
      });

      console.log('Summarize response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Summarize error:', errorText);
        throw new Error(`Failed to summarize: ${response.status}`);
      }

      const data = await response.json();
      console.log('Summarize data:', data);

      if (!data.summary || data.summary.length < 10) {
        throw new Error('Summary too short');
      }

      return { summary: data.summary };
    } catch (error: any) {
      console.error('Summarize error:', error);
      throw error;
    }
  },

  async draftPost(text: string): Promise<{ draft: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/draft`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Draft failed: ${response.status}`);
      }

      return await response.json();
    } catch (error: any) {
      console.error('Draft error:', error);
      throw error;
    }
  },

  async condenseToPost(text: string): Promise<{ draft: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/condense`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Condense failed: ${response.status}`);
      }

      return await response.json();
    } catch (error: any) {
      console.error('Condense error:', error);
      throw error;
    }
  },
};

export default aiService;
