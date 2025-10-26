// src/pages/Index.tsx
import { useState, useEffect } from "react";
import Layout from "@/components/Layout";
import Post from "@/components/Post";
import CreateThread from "@/components/CreateThread";
import ReplyModal from "@/components/ReplyModal";
import aiService from "@/services/aiService";

interface Reply {
  id: string;
  username: string;
  handle: string;
  verified?: boolean;
  time: string;
  content: string;
  images?: string[];
  likes: number;
  views: number;
  isLiked?: boolean;
  sentiment?: { label: string; confidence: number };
}

interface PostData {
  id: string;
  username: string;
  handle: string;
  verified?: boolean;
  time: string;
  content: string;
  images?: string | string[];
  replies: Reply[];
  retweets: number;
  likes: number;
  views: number;
  sentiment?: { label: string; confidence: number };
  flagLabel?: string;
  isLiked?: boolean;
  isRetweeted?: boolean;
  isBookmarked?: boolean;
  showReplies?: boolean;
}

const Index = () => {
  const [posts, setPosts] = useState<PostData[]>([
    {
      id: "1",
      username: "Sarah Chen",
      handle: "sarahchen",
      verified: true,
      time: "2h",
      content: "Just deployed our AI moderation system! Real-time content filtering with high accuracy. The sentiment analysis is working beautifully.",
      replies: [],
      retweets: 45,
      likes: 156,
      views: 3140,
      sentiment: { label: "positive", confidence: 0.95 },
      isLiked: false,
      isRetweeted: false,
      isBookmarked: false,
      showReplies: false,
    },
    {
      id: "2",
      username: "Alex Dev",
      handle: "alexdev",
      verified: false,
      time: "5h",
      content: "Building in public is scary but rewarding. Our hackathon project went from idea to MVP in 18 hours. Key lesson: Ship fast, iterate faster!",
      images: ["https://images.unsplash.com/photo-1551434678-e076c223a692?w=800"],
      replies: [],
      retweets: 234,
      likes: 445,
      views: 8920,
      sentiment: { label: "positive", confidence: 0.89 },
      isLiked: false,
      isRetweeted: false,
      isBookmarked: false,
      showReplies: false,
    },
    {
      id: "3",
      username: "Tech Debates",
      handle: "techdebates",
      verified: true,
      time: "1d",
      content: "Should AI replace human content moderators? Let's discuss the ethics, accuracy, and human oversight needed. What's your take?",
      replies: [
        {
          id: "3-1",
          username: "Maya Singh",
          handle: "mayasingh",
          time: "23h",
          content: "AI should augment, not replace. Humans understand context and nuance that algorithms miss. Hybrid approach is the way forward.",
          likes: 89,
          views: 450,
          sentiment: { label: "neutral", confidence: 0.72 },
        },
        {
          id: "3-2",
          username: "Rob Martinez",
          handle: "robmartinez",
          time: "22h",
          content: "Hard disagree. AI can process millions of posts in seconds. Humans can't scale. We need AI with human oversight for edge cases only.",
          likes: 156,
          views: 890,
          sentiment: { label: "neutral", confidence: 0.68 },
        },
        {
          id: "3-3",
          username: "Dr. Emily Chen",
          handle: "dremilychen",
          verified: true,
          time: "21h",
          content: "From a research perspective, AI has 85-95% accuracy depending on the model. The remaining 5-15% requires human judgment for cultural context, sarcasm, and edge cases. Neither alone is sufficient.",
          likes: 342,
          views: 1200,
          sentiment: { label: "neutral", confidence: 0.81 },
        },
        {
          id: "3-4",
          username: "Jordan Lee",
          handle: "jordanlee",
          time: "20h",
          content: "What about bias in AI training data? If the model learns from biased human decisions, it perpetuates those biases at scale. That's worse than human-only moderation.",
          likes: 201,
          views: 780,
          sentiment: { label: "negative", confidence: 0.65 },
        },
        {
          id: "3-5",
          username: "Priya Sharma",
          handle: "priyasharma",
          time: "19h",
          content: "Valid point about bias. But modern models with proper fairness constraints and diverse training data can actually be LESS biased than individual moderators. It's all about implementation.",
          likes: 178,
          views: 650,
          sentiment: { label: "neutral", confidence: 0.74 },
        },
        {
          id: "3-6",
          username: "Carlos Rivera",
          handle: "carlosrivera",
          time: "18h",
          content: "The real issue is transparency. Users deserve to know why content was flagged or removed. AI systems are black boxes. At least with human moderators, you can appeal and get an explanation.",
          likes: 267,
          views: 920,
          sentiment: { label: "neutral", confidence: 0.70 },
        },
        {
          id: "3-7",
          username: "Aisha Patel",
          handle: "aishapatel",
          time: "17h",
          content: "Explainable AI is a thing now. Models can show which features triggered a decision. Transparency isn't exclusive to humans. Plus, humans make mistakes and have bad days. AI is consistent.",
          likes: 145,
          views: 560,
          sentiment: { label: "neutral", confidence: 0.76 },
        },
        {
          id: "3-8",
          username: "Sam Thompson",
          handle: "samthompson",
          time: "15h",
          content: "Cost is a huge factor too. Hiring and training thousands of moderators globally is expensive. AI can do it for a fraction of the cost. For small platforms, it's the only viable option.",
          likes: 98,
          views: 410,
          sentiment: { label: "neutral", confidence: 0.71 },
        },
        {
          id: "3-9",
          username: "Nina Kowalski",
          handle: "ninakowalski",
          time: "14h",
          content: "But what about the mental health of human moderators who review graphic content daily? AI can handle that trauma without psychological damage. That's a huge advantage.",
          likes: 423,
          views: 1450,
          sentiment: { label: "positive", confidence: 0.79 },
        },
        {
          id: "3-10",
          username: "Tech Debates",
          handle: "techdebates",
          verified: true,
          time: "12h",
          content: "Great discussion everyone! Seems like consensus is emerging: AI for speed and scale, humans for context and appeals, with transparency and fairness as core requirements. The future is collaborative.",
          likes: 512,
          views: 1890,
          sentiment: { label: "positive", confidence: 0.88 },
        },
      ],
      retweets: 89,
      likes: 267,
      views: 5420,
      sentiment: { label: "neutral", confidence: 0.75 },
      isLiked: false,
      isRetweeted: false,
      isBookmarked: false,
      showReplies: false,
    },
  ]);

  const [replyModalOpen, setReplyModalOpen] = useState(false);
  const [replyingTo, setReplyingTo] = useState<PostData | null>(null);

  // AI sentiment analysis for posts and replies on mount
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        // Analyze posts
        const updatedPosts = await Promise.all(
          posts.map(async (p) => {
            if (p.sentiment && p.sentiment.confidence > 0.5) return p;
            try {
              const res = await aiService.analyzePost(p.content);
              return { ...p, sentiment: res.sentiment };
            } catch {
              return p;
            }
          })
        );

        // Analyze replies within posts
        const postsWithAnalyzedReplies = await Promise.all(
          updatedPosts.map(async (post) => {
            if (post.replies.length === 0) return post;

            const analyzedReplies = await Promise.all(
              post.replies.map(async (reply) => {
                if (reply.sentiment && reply.sentiment.confidence > 0.5) return reply;
                try {
                  const res = await aiService.analyzePost(reply.content);
                  return { ...reply, sentiment: res.sentiment };
                } catch {
                  return reply;
                }
              })
            );

            return { ...post, replies: analyzedReplies };
          })
        );

        if (!cancelled) setPosts(postsWithAnalyzedReplies);
      } catch {}
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCreatePost = async (
    content: string,
    image?: string,
    labels?: string[],
    sentiment?: { label: string; confidence: number }
  ) => {
    let flagLabel: string | undefined = undefined;
    if (labels && labels.length > 0) {
      flagLabel = labels[0];
    }

    const newPost: PostData = {
      id: Date.now().toString(),
      username: "Your Name",
      handle: "yourusername",
      verified: false,
      time: "now",
      content,
      images: image ? image : undefined,
      replies: [],
      retweets: 0,
      likes: 0,
      views: 0,
      sentiment,
      flagLabel,
      isLiked: false,
      isRetweeted: false,
      isBookmarked: false,
      showReplies: false,
    };
    setPosts((prev) => [newPost, ...prev]);
  };

  const handleLike = (postId: string) => {
    setPosts((prev) =>
      prev.map((p) =>
        p.id === postId ? { ...p, isLiked: !p.isLiked, likes: p.isLiked ? p.likes - 1 : p.likes + 1 } : p
      )
    );
  };

  const handleReply = (postId: string) => {
    const post = posts.find((p) => p.id === postId);
    if (post) {
      setReplyingTo(post);
      setReplyModalOpen(true);
    }
  };

  const handleSubmitReply = async (content: string, image?: string) => {
    if (!replyingTo) return;

    let sentiment: { label: string; confidence: number } | undefined = undefined;
    try {
      const res = await aiService.analyzePost(content);
      sentiment = res.sentiment;
    } catch {
      sentiment = { label: "neutral", confidence: 0.5 };
    }

    const newReply: Reply = {
      id: Date.now().toString(),
      username: "Your Name",
      handle: "yourusername",
      verified: false,
      time: "now",
      content,
      images: image ? [image] : undefined,
      likes: 0,
      views: 0,
      isLiked: false,
      sentiment,
    };

    setPosts((prev) =>
      prev.map((p) =>
        p.id === replyingTo.id ? { ...p, replies: [...p.replies, newReply], showReplies: true } : p
      )
    );
    setReplyModalOpen(false);
    setReplyingTo(null);
  };

  const handleRetweet = (postId: string) => {
    setPosts((prev) =>
      prev.map((p) =>
        p.id === postId
          ? { ...p, isRetweeted: !p.isRetweeted, retweets: p.isRetweeted ? p.retweets - 1 : p.retweets + 1 }
          : p
      )
    );
  };

  const handleBookmark = (postId: string) => {
    setPosts((prev) =>
      prev.map((p) =>
        p.id === postId ? { ...p, isBookmarked: !p.isBookmarked } : p
      )
    );
  };

  const toggleReplies = (postId: string) => {
    setPosts((prev) => prev.map((p) => (p.id === postId ? { ...p, showReplies: !p.showReplies } : p)));
  };

  return (
    <Layout>
      <CreateThread onPost={handleCreatePost} />

      <div>
        {posts.map((post) => (
          <Post
            key={post.id}
            {...post}
            images={
              post.images
                ? Array.isArray(post.images)
                  ? post.images
                  : [post.images]
                : undefined
            }
            onLike={() => handleLike(post.id)}
            onReply={() => handleReply(post.id)}
            onRetweet={() => handleRetweet(post.id)}
            onBookmark={() => handleBookmark(post.id)}
            onToggleReplies={() => toggleReplies(post.id)}
          />
        ))}
      </div>

      {replyModalOpen && replyingTo && (
        <ReplyModal
          post={{
            username: replyingTo.username,
            handle: replyingTo.handle,
            content: replyingTo.content,
          }}
          onClose={() => setReplyModalOpen(false)}
          onSubmit={handleSubmitReply}
        />
      )}
    </Layout>
  );
};

export default Index;
