// src/pages/Index.tsx
import { useState } from "react";
import Layout from "@/components/Layout";
import Post from "@/components/Post";
import CreateThread from "@/components/CreateThread";
import ReplyModal from "@/components/ReplyModal";

// ---------- Interfaces ----------
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
}

interface PostData {
  id: string;
  username: string;
  handle: string;
  verified?: boolean;
  time: string;
  content: string;
  images?: string[];
  replies: Reply[];
  retweets: number;
  likes: number;
  views: number;
  aiModerated?: boolean;
  aiSummary?: string;
  isLiked?: boolean;
  isRetweeted?: boolean;
  showReplies?: boolean;
}

// ---------- Component ----------
const Index = () => {
  const [posts, setPosts] = useState<PostData[]>([
    {
      id: "1",
      username: "Sarah Chen",
      handle: "sarahchen",
      verified: true,
      time: "2h",
      content:
        "Just deployed our AI moderation system! 🚀 Real-time content filtering with 99.2% accuracy. The sentiment analysis is working beautifully.",
      replies: [],
      retweets: 45,
      likes: 156,
      views: 3140,
      aiModerated: true,
      aiSummary:
        "User announces successful AI moderation deployment with high accuracy for real-time content filtering.",
      isLiked: false,
      isRetweeted: false,
      showReplies: false,
    },
    {
      id: "2",
      username: "Alex Dev",
      handle: "alexdev",
      verified: false,
      time: "5h",
      content:
        "Building in public is scary but rewarding. Our hackathon project went from idea to MVP in 18 hours. Key lesson: Ship fast, iterate faster! 💪",
      images: [
        "https://images.unsplash.com/photo-1551434678-e076c223a692?w=800",
      ],
      replies: [],
      retweets: 234,
      likes: 445,
      views: 8920,
      aiModerated: true,
      aiSummary:
        "Developer shares experience building hackathon MVP in 18 hours, emphasizing rapid iteration.",
      isLiked: false,
      isRetweeted: false,
      showReplies: false,
    },
  ]);

  const [replyModalOpen, setReplyModalOpen] = useState(false);
  const [replyingTo, setReplyingTo] = useState<PostData | null>(null);

  // ---------- Handlers ----------

  const handleCreatePost = (content: string, image?: string) => {
    const newPost: PostData = {
      id: Date.now().toString(),
      username: "Your Name",
      handle: "yourusername",
      verified: false,
      time: "now",
      content,
      images: image ? [image] : undefined,
      replies: [],
      retweets: 0,
      likes: 0,
      views: 0,
      aiModerated: true,
      aiSummary: "User shared a new post on the platform.",
      isLiked: false,
      isRetweeted: false,
      showReplies: false,
    };
    setPosts([newPost, ...posts]);
  };

  const handleLike = (postId: string) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === postId
          ? {
              ...post,
              isLiked: !post.isLiked,
              likes: post.isLiked ? post.likes - 1 : post.likes + 1,
            }
          : post
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

  const handleSubmitReply = (content: string, image?: string) => {
    if (!replyingTo) return;

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
    };

    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === replyingTo.id
          ? {
              ...post,
              replies: [...post.replies, newReply],
              showReplies: true,
            }
          : post
      )
    );

    setReplyModalOpen(false);
    setReplyingTo(null);
  };

  const handleRetweet = (postId: string) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === postId
          ? {
              ...post,
              isRetweeted: !post.isRetweeted,
              retweets: post.isRetweeted
                ? post.retweets - 1
                : post.retweets + 1,
            }
          : post
      )
    );
  };

  const toggleReplies = (postId: string) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === postId
          ? { ...post, showReplies: !post.showReplies }
          : post
      )
    );
  };

  // ---------- Render ----------
  return (
    <Layout>
      {/* CreateThread input */}
      <CreateThread onPost={handleCreatePost} />

      {/* Feed */}
      <div>
        {posts.map((post) => (
          <Post
            key={post.id}
            {...post}
            onLike={() => handleLike(post.id)}
            onReply={() => handleReply(post.id)}
            onRetweet={() => handleRetweet(post.id)}
            onToggleReplies={() => toggleReplies(post.id)}
          />
        ))}
      </div>

      {/* Reply Modal */}
      {replyModalOpen && replyingTo && (
        <ReplyModal
          post={replyingTo}
          onClose={() => setReplyModalOpen(false)}
          onSubmit={handleSubmitReply}
        />
      )}
    </Layout>
  );
};

export default Index;
