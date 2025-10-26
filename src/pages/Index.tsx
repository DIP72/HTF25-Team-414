import { useState, useEffect } from "react";
import Layout from "@/components/Layout";
import Post from "@/components/Post";
import CreateThread from "@/components/CreateThread";
import ReplyModal from "@/components/ReplyModal";
import aiService from "@/services/aiService";
import { useAuth } from "@/contexts/AuthContext";

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
  const { profile } = useAuth();

  const CURRENT_USER = {
    username: profile?.username || "User",
    handle: profile?.handle || "user",
    verified: profile?.verified || false,
    isAdmin: profile?.role === "admin",
  };

  const [posts, setPosts] = useState<PostData[]>([
    {
      id: "1",
      username: "Sarah Chen",
      handle: "sarahchen",
      verified: true,
      time: "2h",
      content:
        "Just deployed our AI moderation system! Real-time content filtering with high accuracy. The sentiment analysis is working beautifully.",
      replies: [],
      retweets: 45,
      likes: 156,
      views: 3140,
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
      content:
        "Building in public is scary but rewarding. Our hackathon project went from idea to MVP in 18 hours. Key lesson: Ship fast, iterate faster!",
      images: [
        "https://images.unsplash.com/photo-1551434678-e076c223a692?w=800",
      ],
      replies: [],
      retweets: 234,
      likes: 445,
      views: 8920,
      isLiked: false,
      isRetweeted: false,
      isBookmarked: false,
      showReplies: false,
    },
    // ...other posts
  ]);

  const [replyModalOpen, setReplyModalOpen] = useState(false);
  const [replyingTo, setReplyingTo] = useState<PostData | null>(null);

  // Analyze sentiment for all posts & replies on mount
  useEffect(() => {
    const analyzeSentiment = async () => {
      const updatedPosts = await Promise.all(
        posts.map(async (post) => {
          try {
            const { sentiment } = await aiService.getSentimentOnly(post.content);

            const updatedReplies = await Promise.all(
              post.replies.map(async (reply) => {
                try {
                  const { sentiment } = await aiService.getSentimentOnly(reply.content);
                  return { ...reply, sentiment };
                } catch {
                  return reply;
                }
              })
            );

            return { ...post, sentiment, replies: updatedReplies };
          } catch {
            return post;
          }
        })
      );
      setPosts(updatedPosts);
    };

    analyzeSentiment();
  }, []);

  const handleCreatePost = async (
    content: string,
    images?: string[],
    labels?: string[],
    sentiment?: { label: string; confidence: number }
  ) => {
    let finalSentiment = sentiment;

    if (!finalSentiment) {
      try {
        const result = await aiService.getSentimentOnly(content);
        finalSentiment = result.sentiment;
      } catch {
        finalSentiment = { label: "NEUTRAL", confidence: 0.5 };
      }
    }

    const newPost: PostData = {
      id: Date.now().toString(),
      username: CURRENT_USER.username,
      handle: CURRENT_USER.handle,
      verified: CURRENT_USER.verified,
      time: "now",
      content,
      images: images && images.length > 0 ? images : undefined,
      replies: [],
      retweets: 0,
      likes: 0,
      views: 0,
      sentiment: finalSentiment,
      flagLabel: labels && labels.length > 0 ? labels[0] : undefined,
      isLiked: false,
      isRetweeted: false,
      isBookmarked: false,
      showReplies: false,
    };

    setPosts((prev) => [newPost, ...prev]);
  };

  const handleDeletePost = (postId: string) =>
    setPosts((prev) => prev.filter((p) => p.id !== postId));

  const handleEditPost = (postId: string, newContent: string) =>
    setPosts((prev) =>
      prev.map((p) => (p.id === postId ? { ...p, content: newContent } : p))
    );

  const handleLike = (postId: string) =>
    setPosts((prev) =>
      prev.map((p) =>
        p.id === postId
          ? { ...p, isLiked: !p.isLiked, likes: p.isLiked ? p.likes - 1 : p.likes + 1 }
          : p
      )
    );

  const handleReplyLike = (postId: string, replyId: string) =>
    setPosts((prev) =>
      prev.map((p) => {
        if (p.id === postId) {
          return {
            ...p,
            replies: p.replies.map((r) =>
              r.id === replyId
                ? { ...r, isLiked: !r.isLiked, likes: r.isLiked ? r.likes - 1 : r.likes + 1 }
                : r
            ),
          };
        }
        return p;
      })
    );

  const handleReply = (postId: string) => {
    const post = posts.find((p) => p.id === postId);
    if (post) {
      setReplyingTo(post);
      setReplyModalOpen(true);
    }
  };

  const handleSubmitReply = async (content: string, images?: string[]) => {
    if (!replyingTo) return;

    let sentiment: { label: string; confidence: number } | undefined = undefined;
    try {
      const res = await aiService.getSentimentOnly(content);
      sentiment = res.sentiment;
    } catch {
      sentiment = { label: "NEUTRAL", confidence: 0.5 };
    }

    const newReply: Reply = {
      id: Date.now().toString(),
      username: CURRENT_USER.username,
      handle: CURRENT_USER.handle,
      verified: CURRENT_USER.verified,
      time: "now",
      content,
      images: images && images.length > 0 ? images : undefined,
      likes: 0,
      views: 0,
      isLiked: false,
      sentiment,
    };

    setPosts((prev) =>
      prev.map((p) =>
        p.id === replyingTo.id
          ? { ...p, replies: [...p.replies, newReply], showReplies: true }
          : p
      )
    );

    setReplyModalOpen(false);
    setReplyingTo(null);
  };

  const handleRetweet = (postId: string) =>
    setPosts((prev) =>
      prev.map((p) =>
        p.id === postId
          ? { ...p, isRetweeted: !p.isRetweeted, retweets: p.isRetweeted ? p.retweets - 1 : p.retweets + 1 }
          : p
      )
    );

  const handleBookmark = (postId: string) =>
    setPosts((prev) =>
      prev.map((p) =>
        p.id === postId ? { ...p, isBookmarked: !p.isBookmarked } : p
      )
    );

  const toggleReplies = (postId: string) =>
    setPosts((prev) =>
      prev.map((p) => (p.id === postId ? { ...p, showReplies: !p.showReplies } : p))
    );

  return (
    <Layout>
      <CreateThread onPost={handleCreatePost} currentUser={CURRENT_USER} />
      <div>
        {posts.map((post) => (
          <Post
            key={post.id}
            {...post}
            images={post.images ? (Array.isArray(post.images) ? post.images : [post.images]) : undefined}
            currentUser={CURRENT_USER}
            onLike={() => handleLike(post.id)}
            onReply={() => handleReply(post.id)}
            onRetweet={() => handleRetweet(post.id)}
            onBookmark={() => handleBookmark(post.id)}
            onToggleReplies={() => toggleReplies(post.id)}
            onReplyLike={(replyId) => handleReplyLike(post.id, replyId)}
            onDelete={() => handleDeletePost(post.id)}
            onEdit={(newContent) => handleEditPost(post.id, newContent)}
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
          currentUser={CURRENT_USER}
          onClose={() => setReplyModalOpen(false)}
          onSubmit={handleSubmitReply}
        />
      )}
    </Layout>
  );
};

export default Index;
