import { useState, useEffect } from "react";
import Layout from "@/components/Layout";
import Post from "@/components/Post";
import CreateThread from "@/components/CreateThread";
import ReplyModal from "@/components/ReplyModal";
import aiService from "@/services/aiService";
import { postsService } from "@/services/postsService";
import { supabase } from "@/lib/supabase";
import { useAuth } from "@/contexts/AuthContext";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

interface Reply {
  id: string;
  username: string;
  handle: string;
  role?: 'user' | 'admin' | 'verified';
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
  role?: 'user' | 'admin' | 'verified';
  time: string;
  content: string;
  images?: string | string[];
  replies: Reply[];
  reposts: number;
  likes: number;
  views: number;
  sentiment?: { label: string; confidence: number };
  flagLabel?: string;
  isLiked?: boolean;
  isReposted?: boolean;
  isBookmarked?: boolean;
  showReplies?: boolean;
}

const Index = () => {
  const { profile, user } = useAuth();

  const CURRENT_USER = {
    username: profile?.username || "User",
    handle: profile?.handle || "user",
    verified: profile?.role === 'verified' || profile?.role === 'admin',
    isAdmin: profile?.role === "admin",
  };

  const [posts, setPosts] = useState<PostData[]>([]);
  const [loading, setLoading] = useState(true);
  const [replyModalOpen, setReplyModalOpen] = useState(false);
  const [replyingTo, setReplyingTo] = useState<PostData | null>(null);

  useEffect(() => {
    loadPosts();

    const channel = postsService.subscribeToPostUpdates((payload) => {
      console.log('Real-time update:', payload);
      if (payload.eventType === 'INSERT') {
        loadPosts();
      }
    });

    return () => {
      supabase.removeChannel(channel);
    };
  }, [user]);

  const loadPosts = async () => {
    setLoading(true);
    try {
      const fetchedPosts = await postsService.fetchPosts(user?.id);

      const formattedPosts: PostData[] = await Promise.all(
        fetchedPosts.map(async (post) => {
          const replies = await postsService.fetchReplies(post.id);
          
          const formattedReplies: Reply[] = replies.map(reply => ({
            id: reply.id,
            username: reply.username,
            handle: reply.handle,
            role: reply.role,
            time: formatTimeAgo(reply.created_at),
            content: reply.content,
            images: reply.images || undefined,
            likes: reply.likes_count,
            views: reply.views_count,
            isLiked: false,
            sentiment: reply.sentiment_label ? {
              label: reply.sentiment_label,
              confidence: reply.sentiment_confidence || 0
            } : undefined,
          }));

          return {
            id: post.id,
            username: post.username,
            handle: post.handle,
            role: post.role,
            time: formatTimeAgo(post.created_at),
            content: post.content,
            images: post.images || undefined,
            replies: formattedReplies,
            reposts: post.reposts_count,
            likes: post.likes_count,
            views: post.views_count,
            sentiment: post.sentiment_label ? {
              label: post.sentiment_label,
              confidence: post.sentiment_confidence || 0
            } : undefined,
            flagLabel: post.flag_label || undefined,
            isLiked: post.isLiked,
            isReposted: post.isReposted,
            isBookmarked: post.isBookmarked,
            showReplies: false,
          };
        })
      );

      setPosts(formattedPosts);
    } catch (error) {
      console.error('Error loading posts:', error);
      toast.error('Failed to load posts');
    } finally {
      setLoading(false);
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d`;
    return date.toLocaleDateString();
  };

  const handleCreatePost = async (
    content: string,
    images?: string[],
    labels?: string[],
    sentiment?: { label: string; confidence: number }
  ) => {
    try {
      const post = await postsService.createPost({
        content,
        images,
        sentiment_label: sentiment?.label,
        sentiment_confidence: sentiment?.confidence,
        flag_label: labels && labels.length > 0 ? labels[0] : undefined,
      });

      if (post) {
        await loadPosts();
      }
    } catch (error) {
      console.error('Failed to create post:', error);
    }
  };

  const handleDeletePost = async (postId: string) => {
    setPosts(prev => prev.filter(p => p.id !== postId));
    
    const success = await postsService.deletePost(postId);
    if (!success) {
      loadPosts();
    }
  };

  const handleEditPost = async (postId: string, newContent: string) => {
    setPosts(prev => prev.map(p => 
      p.id === postId ? { ...p, content: newContent } : p
    ));
    
    const success = await postsService.updatePost(postId, newContent);
    if (!success) {
      loadPosts();
    }
  };

  const handleLike = async (postId: string) => {
    // Optimistic update
    setPosts(prev => prev.map(p => {
      if (p.id === postId) {
        const newIsLiked = !p.isLiked;
        return {
          ...p,
          isLiked: newIsLiked,
          likes: newIsLiked ? p.likes + 1 : p.likes - 1,
        };
      }
      return p;
    }));
    
    const success = await postsService.toggleLike(postId);
    
    if (success) {
      // Sync with DB after trigger runs
      setTimeout(async () => {
        try {
          const realCount = await postsService.getLikeCount(postId);
          
          if (user) {
            const { data: userLike } = await supabase
              .from('likes')
              .select('id')
              .eq('user_id', user.id)
              .eq('post_id', postId)
              .maybeSingle();
            
            setPosts(prev => prev.map(p => 
              p.id === postId ? { 
                ...p, 
                likes: realCount,
                isLiked: !!userLike 
              } : p
            ));
          }
        } catch (error) {
          console.error('Failed to sync like count:', error);
        }
      }, 500);
    }
  };

  const handleRepost = async (postId: string) => {
    // Optimistic update
    setPosts(prev => prev.map(p => {
      if (p.id === postId) {
        const newIsReposted = !p.isReposted;
        return {
          ...p,
          isReposted: newIsReposted,
          reposts: newIsReposted ? p.reposts + 1 : p.reposts - 1,
        };
      }
      return p;
    }));
    
    const success = await postsService.toggleRepost(postId);
    
    if (success) {
      // Sync with DB after trigger runs
      setTimeout(async () => {
        try {
          const realCount = await postsService.getRepostCount(postId);
          
          if (user) {
            const { data: userRepost } = await supabase
              .from('reposts')
              .select('id')
              .eq('user_id', user.id)
              .eq('post_id', postId)
              .maybeSingle();
            
            setPosts(prev => prev.map(p => 
              p.id === postId ? { 
                ...p, 
                reposts: realCount,
                isReposted: !!userRepost 
              } : p
            ));
          }
        } catch (error) {
          console.error('Failed to sync repost count:', error);
        }
      }, 500);
    }
  };

  const handleReplyLike = async (postId: string, replyId: string) => {
    setPosts(prev => prev.map(p => {
      if (p.id === postId) {
        return {
          ...p,
          replies: p.replies.map(r => {
            if (r.id === replyId) {
              const newIsLiked = !r.isLiked;
              return {
                ...r,
                isLiked: newIsLiked,
                likes: newIsLiked ? r.likes + 1 : r.likes - 1,
              };
            }
            return r;
          }),
        };
      }
      return p;
    }));
  };

  const handleReply = (postId: string) => {
    const post = posts.find((p) => p.id === postId);
    if (post) {
      setReplyingTo(post);
      setReplyModalOpen(true);
    }
  };

  const handleSubmitReply = async (content: string, images?: string[]) => {
    if (!replyingTo) return;

    try {
      let sentiment: { label: string; confidence: number } | undefined = undefined;
      try {
        const res = await aiService.getSentimentOnly(content);
        sentiment = res.sentiment;
      } catch {
        sentiment = { label: "NEUTRAL", confidence: 0.5 };
      }

      const newReply = await postsService.createReply(
        replyingTo.id,
        content,
        sentiment?.label,
        sentiment?.confidence
      );

      if (newReply) {
        const formattedReply: Reply = {
          id: newReply.id,
          username: newReply.username,
          handle: newReply.handle,
          role: newReply.role,
          time: 'now',
          content: newReply.content,
          images: newReply.images || undefined,
          likes: 0,
          views: 0,
          isLiked: false,
          sentiment,
        };

        setPosts(prev => prev.map(p => {
          if (p.id === replyingTo.id) {
            return {
              ...p,
              replies: [...p.replies, formattedReply],
              showReplies: true,
            };
          }
          return p;
        }));
      }
    } catch (error) {
      console.error('Error posting reply:', error);
    }

    setReplyModalOpen(false);
    setReplyingTo(null);
  };

  const handleBookmark = async (postId: string) => {
    setPosts(prev => prev.map(p => 
      p.id === postId ? { ...p, isBookmarked: !p.isBookmarked } : p
    ));
    
    await postsService.toggleBookmark(postId);
  };

  const toggleReplies = (postId: string) => {
    setPosts(prev => prev.map(p => 
      p.id === postId ? { ...p, showReplies: !p.showReplies } : p
    ));
  };

  return (
    <Layout>
      <CreateThread onPost={handleCreatePost} currentUser={CURRENT_USER} />

      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        </div>
      )}

      {!loading && posts.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No posts yet. Be the first to post!</p>
        </div>
      )}

      <div>
        {!loading && posts.map((post) => (
          <Post
            key={post.id}
            {...post}
            images={post.images ? (Array.isArray(post.images) ? post.images : [post.images]) : undefined}
            currentUser={CURRENT_USER}
            onLike={() => handleLike(post.id)}
            onReply={() => handleReply(post.id)}
            onRepost={() => handleRepost(post.id)}
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
