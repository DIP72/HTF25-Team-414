import { Heart, MessageCircle, Repeat2, Bookmark, MoreHorizontal, User, BarChart, X, ChevronLeft, ChevronRight, Sparkles } from "lucide-react";
import { useState } from "react";
import aiService from "@/services/aiService";
import { parseMarkdown } from "@/utils/markdown";

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

interface PostProps {
  id?: string;
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
  sentiment?: { label: string; confidence: number };
  flagLabel?: string;
  isLiked?: boolean;
  isRetweeted?: boolean;
  isBookmarked?: boolean;
  showReplies?: boolean;
  onLike?: () => void;
  onReply?: () => void;
  onRetweet?: () => void;
  onBookmark?: () => void;
  onToggleReplies?: () => void;
}

const Post = (props: PostProps) => {
  const {
    username, handle, verified, time, content, images, replies,
    retweets, likes, views, sentiment, flagLabel,
    isLiked = false, isRetweeted = false, isBookmarked = false, showReplies = false,
    onLike, onReply, onRetweet, onBookmark, onToggleReplies
  } = props;

  const [imageModalOpen, setImageModalOpen] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [threadSummary, setThreadSummary] = useState<string | null>(null);

  const openImageModal = (index: number) => { setCurrentImageIndex(index); setImageModalOpen(true); };
  const nextImage = () => { if (images) setCurrentImageIndex((p) => (p + 1) % images.length); };
  const prevImage = () => { if (images) setCurrentImageIndex((p) => (p - 1 + images.length) % images.length); };

  const summarizeThisThread = async () => {
    if (isSummarizing) return;
    setIsSummarizing(true);
    try {
      const texts = [content, ...replies.map((r) => r.content)];
      const image_counts = [(images?.length ?? 0), ...replies.map((r) => r.images?.length ?? 0)];
      const image_alts: string[] = [];
      const { summary } = await aiService.summarizeThread({ texts, image_counts, image_alts });
      setThreadSummary(summary || "");
    } catch {
      setThreadSummary("Summary unavailable");
    } finally {
      setIsSummarizing(false);
    }
  };

  return (
    <div className="mb-4">
      <article className="bg-gray-50 rounded-2xl p-3 sm:p-4 hover:bg-gray-100 transition-colors duration-200">
        <div className="flex gap-2 sm:gap-3">
          <div className="relative flex-shrink-0">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-gray-300 flex items-center justify-center">
              <User className="w-5 h-5 sm:w-6 sm:h-6 text-gray-600" strokeWidth={2} />
            </div>
          </div>

          <div className="flex-1 min-w-0">
            {/* Header */}
            <div className="mb-1">
              <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
                <span className="font-bold text-sm sm:text-[15px] text-gray-900 hover:underline cursor-pointer truncate">{username}</span>
                {verified && (
                  <svg width="16" height="16" viewBox="0 0 22 22" className="text-blue-500 flex-shrink-0">
                    <path fill="currentColor" d="M20.396 11c-.018-.646-.215-1.275-.57-1.816-.354-.54-.852-.972-1.438-1.246.223-.607.27-1.264.14-1.897-.131-.634-.437-1.218-.882-1.687-.47-.445-1.053-.75-1.687-.882-.633-.13-1.29-.083-1.897.14-.273-.587-.704-1.086-1.245-1.44S11.647 1.62 11 1.604c-.646.017-1.273.213-1.813.568s-.969.854-1.24 1.44c-.608-.223-1.267-.272-1.902-.14-.635.13-1.22.436-1.69.882-.445.47-.749 1.055-.878 1.688-.13.633-.08 1.29.144 1.896-.587.274-1.087.705-1.443 1.245-.356.54-.555 1.17-.574 1.817.02.647.218 1.276.574 1.817.356.54.856.972 1.443 1.245-.224.606-.274 1.263-.144 1.896.13.634.433 1.218.877 1.688.47.443 1.054.747 1.687.878.633.132 1.29.084 1.897-.136.274.586.705 1.084 1.246 1.439.54.354 1.17.551 1.816.569.647-.016 1.276-.213 1.817-.567s.972-.854 1.245-1.44c.604.239 1.266.296 1.903.164.636-.132 1.22-.447 1.68-.907.46-.46.776-1.044.908-1.681s.075-1.299-.165-1.903c.586-.274 1.084-.705 1.439-1.246.354-.54.551-1.17.569-1.816zM9.662 14.85l-3.429-3.428 1.293-1.302 2.072 2.072 4.4-4.794 1.347 1.246z"/>
                  </svg>
                )}
                <span className="text-xs sm:text-[15px] text-gray-500 truncate">@{handle}</span>
                <span className="text-gray-500 hidden sm:inline">·</span>
                <span className="text-xs sm:text-[15px] text-gray-500">{time}</span>
                <button className="ml-auto text-gray-500 hover:bg-gray-200 hover:text-gray-900 rounded-full p-1.5 sm:p-2 transition-all duration-200">
                  <MoreHorizontal className="w-4 h-4 sm:w-5 sm:h-5" />
                </button>
              </div>

              {/* Sentiment & Flag */}
              {(sentiment || flagLabel) && (
                <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                  {sentiment && (
                    <span className={`text-[10px] sm:text-[11px] px-2 sm:px-2.5 py-0.5 rounded-full font-medium whitespace-nowrap ${
                      sentiment.label.toLowerCase() === 'positive' ? 'bg-green-100 text-green-700'
                      : sentiment.label.toLowerCase() === 'negative' ? 'bg-red-100 text-red-700'
                      : 'bg-gray-100 text-gray-700'
                    }`}>
                      {sentiment.label.charAt(0).toUpperCase() + sentiment.label.slice(1)} • {Math.round(sentiment.confidence * 100)}%
                    </span>
                  )}
                  {flagLabel && (
                    <span className="text-[10px] sm:text-[11px] px-2 sm:px-2.5 py-0.5 rounded-full font-medium bg-amber-100 text-amber-700 whitespace-nowrap">
                      {flagLabel}
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Content */}
            <div className="text-sm sm:text-[15px] text-gray-900 mb-2 sm:mb-3 leading-normal break-words whitespace-pre-wrap">
              {parseMarkdown(content)}
            </div>

            {/* Thread Summary */}
            {threadSummary && (
              <div className="mb-2 sm:mb-3 w-full">
                <div className="inline-flex max-w-full bg-blue-100 rounded-2xl px-3 py-2">
                  <p className="text-[11px] sm:text-[12px] leading-relaxed text-blue-700 break-words whitespace-normal overflow-wrap-anywhere">
                    {threadSummary}
                  </p>
                </div>
              </div>
            )}

            {/* Images */}
            {images && images.length > 0 && (
              <div className={`mb-2 sm:mb-3 rounded-2xl overflow-hidden cursor-pointer ${
                images.length === 1 ? "grid grid-cols-1" : "grid grid-cols-2 gap-1 sm:gap-2"
              }`}>
                {images.map((img, idx) => (
                  <div 
                    key={idx} 
                    className={`relative overflow-hidden bg-gray-200 rounded-lg sm:rounded-xl hover:opacity-90 transition-opacity ${
                      images.length === 1 ? "aspect-video" : images.length === 3 && idx === 0 ? "col-span-2 aspect-video" : "aspect-square"
                    }`} 
                    onClick={() => openImageModal(idx)}
                  >
                    <img src={img} alt="" className="w-full h-full object-cover" />
                    {images.length > 1 && (
                      <div className="absolute top-1 right-1 sm:top-2 sm:right-2 bg-black/60 text-white text-[10px] sm:text-xs px-1.5 sm:px-2 py-0.5 sm:py-1 rounded-full">
                        {idx + 1}/{images.length}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Actions (like, reply, retweet, bookmark) */}
            <div className="flex items-center justify-between mt-2 sm:mt-3">
              <div className="flex items-center gap-1 sm:gap-2">
                <button onClick={onReply} className="group flex items-center gap-0.5 sm:gap-1 text-gray-500 hover:text-gray-900 transition-colors">
                  <div className="p-1.5 sm:p-2 rounded-full group-hover:bg-gray-200 transition-colors">
                    <MessageCircle className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={2} />
                  </div>
                  <span className="text-xs sm:text-sm">{replies.length}</span>
                </button>

                <button onClick={onRetweet} className={`group flex items-center gap-0.5 sm:gap-1 transition-colors ${isRetweeted ? "text-green-600" : "text-gray-500 hover:text-gray-900"}`}>
                  <div className={`p-1.5 sm:p-2 rounded-full transition-colors ${isRetweeted ? "bg-green-50" : "group-hover:bg-gray-200"}`}>
                    <Repeat2 className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={2} />
                  </div>
                  <span className="text-xs sm:text-sm">{retweets}</span>
                </button>

                <button onClick={onLike} className={`group flex items-center gap-0.5 sm:gap-1 transition-colors ${isLiked ? "text-pink-600" : "text-gray-500 hover:text-gray-900"}`}>
                  <div className={`p-1.5 sm:p-2 rounded-full transition-colors ${isLiked ? "bg-pink-50" : "group-hover:bg-gray-200"}`}>
                    <Heart className={`w-4 h-4 sm:w-5 sm:h-5 ${isLiked ? "fill-current" : ""}`} strokeWidth={2} />
                  </div>
                  <span className="text-xs sm:text-sm">{likes}</span>
                </button>

                <button className="group flex items-center gap-0.5 sm:gap-1 text-gray-500 hover:text-gray-900 transition-colors">
                  <div className="p-1.5 sm:p-2 rounded-full group-hover:bg-gray-200 transition-colors">
                    <BarChart className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={2} />
                  </div>
                  <span className="text-xs sm:text-sm">{views}</span>
                </button>
              </div>

              <div className="flex items-center gap-1">
                <button onClick={summarizeThisThread} disabled={isSummarizing} className="group flex items-center gap-1 text-blue-600 hover:bg-blue-50 rounded-full px-2 sm:px-3 py-1 transition" title="Summarize this thread">
                  <Sparkles className={`w-3 h-3 sm:w-4 sm:h-4 ${isSummarizing ? "animate-pulse" : ""}`} />
                  <span className="text-[10px] sm:text-xs font-semibold hidden sm:inline">{isSummarizing ? "Summarizing..." : "Summarize"}</span>
                </button>

                <button onClick={onBookmark} className={`group transition-colors ${isBookmarked ? "text-blue-600" : "text-gray-500 hover:text-gray-900"}`} title={isBookmarked ? "Remove bookmark" : "Bookmark"}>
                  <div className={`p-2 rounded-full transition-colors ${isBookmarked ? "bg-blue-50" : "group-hover:bg-gray-200"}`}>
                    <Bookmark className={`w-5 h-5 ${isBookmarked ? "fill-current" : ""}`} strokeWidth={2} />
                  </div>
                </button>
              </div>
            </div>

            {/* Replies */}
            {replies.length > 0 && (
              <div className="mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-gray-200">
                <button onClick={onToggleReplies} className="text-xs sm:text-sm text-gray-900 hover:underline font-medium mb-2 sm:mb-3">
                  {showReplies ? `Hide ${replies.length} ${replies.length === 1 ? "reply" : "replies"}` : `View ${replies.length} ${replies.length === 1 ? "reply" : "replies"}`}
                </button>
                {showReplies && (
                  <div className="space-y-2 sm:space-y-3 mt-2 sm:mt-3">
                    {replies.map((reply) => (
                      <div key={reply.id} className="flex gap-2 sm:gap-3 pl-2 sm:pl-3 border-l-2 border-gray-300">
                        <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                          <User className="w-3 h-3 sm:w-4 sm:h-4 text-gray-600" strokeWidth={2} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5 sm:gap-2 mb-0.5 sm:mb-1 flex-wrap">
                            <span className="font-bold text-xs sm:text-sm text-gray-900 truncate">{reply.username}</span>
                            <span className="text-xs sm:text-sm text-gray-500 truncate">@{reply.handle}</span>
                            <span className="text-gray-500 hidden sm:inline">·</span>
                            <span className="text-xs sm:text-sm text-gray-500">{reply.time}</span>
                            {reply.sentiment && (
                              <span className={`text-[9px] sm:text-[10px] px-1.5 py-0.5 rounded-full font-medium whitespace-nowrap ${
                                reply.sentiment.label.toLowerCase() === 'positive' ? 'bg-green-100 text-green-700'
                                : reply.sentiment.label.toLowerCase() === 'negative' ? 'bg-red-100 text-red-700'
                                : 'bg-gray-100 text-gray-700'
                              }`}>
                                {reply.sentiment.label.charAt(0).toUpperCase() + reply.sentiment.label.slice(1)} • {Math.round(reply.sentiment.confidence * 100)}%
                              </span>
                            )}
                          </div>
                          <div className="text-xs sm:text-sm text-gray-900 mb-1 sm:mb-2 break-words whitespace-pre-wrap">
                            {parseMarkdown(reply.content)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

          </div>
        </div>
      </article>

      {/* Image modal */}
      {imageModalOpen && images && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-2 sm:p-4" onClick={() => setImageModalOpen(false)}>
          <button onClick={() => setImageModalOpen(false)} className="absolute top-2 right-2 sm:top-4 sm:right-4 text-white hover:bg-white/20 rounded-full p-1.5 sm:p-2 transition-colors z-10">
            <X className="w-5 h-5 sm:w-6 sm:h-6" />
          </button>
          <div className="relative max-w-4xl max-h-[90vh] w-full" onClick={(e) => e.stopPropagation()}>
            <img src={images[currentImageIndex]} alt="" className="w-full h-full object-contain rounded-lg" />
            {images.length > 1 && (
              <>
                <button onClick={prevImage} className="absolute left-2 sm:left-4 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full p-2 sm:p-3 transition-colors">
                  <ChevronLeft className="w-5 h-5 sm:w-6 sm:h-6" />
                </button>
                <button onClick={nextImage} className="absolute right-2 sm:right-4 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full p-2 sm:p-3 transition-colors">
                  <ChevronRight className="w-5 h-5 sm:w-6 sm:h-6" />
                </button>
                <div className="absolute bottom-2 sm:bottom-4 left-1/2 -translate-x-1/2 bg-black/60 text-white text-xs sm:text-sm px-3 sm:px-4 py-1.5 sm:py-2 rounded-full">
                  {currentImageIndex + 1} / {images.length}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Post;
