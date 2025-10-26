import { useState } from "react";
import { 
  Heart, 
  MessageCircle, 
  Repeat2, 
  Bookmark, 
  BarChart, 
  Share, 
  ChevronDown, 
  ChevronUp,
  User,
  X,
  ChevronLeft,
  ChevronRight
} from "lucide-react";
import { parseMarkdown } from "@/utils/markdown";

interface Sentiment {
  label: string;
  confidence: number;
}

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
  sentiment?: Sentiment;
}

interface PostProps {
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
  sentiment?: Sentiment;
  flagLabel?: string;
  currentUser: {
    username: string;
    handle: string;
    verified: boolean;
    isAdmin: boolean;
  };
  isLiked?: boolean;
  isRetweeted?: boolean;
  isBookmarked?: boolean;
  showReplies?: boolean;
  onLike?: () => void;
  onReply?: () => void;
  onRetweet?: () => void;
  onBookmark?: () => void;
  onToggleReplies?: () => void;
  onReplyLike?: (replyId: string) => void;
  onDelete?: () => void;
  onEdit?: (newContent: string) => void;
}

const Post = ({
  id,
  username,
  handle,
  verified,
  time,
  content,
  images = [],
  replies,
  retweets,
  likes,
  views,
  sentiment,
  flagLabel,
  currentUser,
  isLiked,
  isRetweeted,
  isBookmarked,
  showReplies = false,
  onLike,
  onReply,
  onRetweet,
  onBookmark,
  onToggleReplies,
  onReplyLike,
  onDelete,
  onEdit,
}: PostProps) => {
  const [imageModalOpen, setImageModalOpen] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const nextImage = () => {
    if (images && images.length > 0) {
      setCurrentImageIndex((prev) => (prev + 1) % images.length);
    }
  };

  const prevImage = () => {
    if (images && images.length > 0) {
      setCurrentImageIndex((prev) => 
        prev === 0 ? images.length - 1 : prev - 1
      );
    }
  };

  const getSentimentColor = (label: string) => {
    switch (label.toLowerCase()) {
      case "positive":
        return "bg-green-100 text-green-700 border-green-200";
      case "negative":
        return "bg-red-100 text-red-700 border-red-200";
      default:
        return "bg-gray-100 text-gray-700 border-gray-200";
    }
  };

  const getFlagColor = (flag: string) => {
    if (flag.toLowerCase().includes("block")) {
      return "bg-red-100 text-red-700 border-red-300";
    } else if (flag.toLowerCase().includes("flag")) {
      return "bg-amber-100 text-amber-700 border-amber-300";
    }
    return "bg-gray-100 text-gray-700 border-gray-300";
  };

  return (
    <article className="bg-white border-b border-gray-200 hover:bg-gray-50 transition-colors">
      <div className="p-4">
        {/* Header */}
        <div className="flex gap-3">
          {/* Avatar */}
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
            <User className="w-5 h-5 sm:w-6 sm:h-6 text-gray-600" strokeWidth={2} />
          </div>

          <div className="flex-1 min-w-0">
            {/* User Info */}
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <span className="font-bold text-sm sm:text-base text-gray-900 truncate">
                {username}
              </span>
              
              {verified && (
                <svg width="18" height="18" viewBox="0 0 22 22" className="text-blue-500 flex-shrink-0">
                  <path
                    fill="currentColor"
                    d="M20.396 11c-.018-.646-.215-1.275-.57-1.816-.354-.54-.852-.972-1.438-1.246.223-.607.27-1.264.14-1.897-.131-.634-.437-1.218-.882-1.687-.47-.445-1.053-.75-1.687-.882-.633-.13-1.29-.083-1.897.14-.273-.587-.704-1.086-1.245-1.44S11.647 1.62 11 1.604c-.646.017-1.273.213-1.813.568s-.969.854-1.24 1.44c-.608-.223-1.267-.272-1.902-.14-.635.13-1.22.436-1.69.882-.445.47-.749 1.055-.878 1.688-.13.633-.08 1.29.144 1.896-.587.274-1.087.705-1.443 1.245-.356.54-.555 1.17-.574 1.817.02.647.218 1.276.574 1.817.356.54.856.972 1.443 1.245-.224.606-.274 1.263-.144 1.896.13.634.433 1.218.877 1.688.47.443 1.054.747 1.687.878.633.132 1.29.084 1.897-.136.274.586.705 1.084 1.246 1.439.54.354 1.17.551 1.816.569.647-.016 1.276-.213 1.817-.567s.972-.854 1.245-1.44c.604.239 1.266.296 1.903.164.636-.132 1.22-.447 1.68-.907.46-.46.776-1.044.908-1.681s.075-1.299-.165-1.903c.586-.274 1.084-.705 1.439-1.246.354-.54.551-1.17.569-1.816zM9.662 14.85l-3.429-3.428 1.293-1.302 2.072 2.072 4.4-4.794 1.347 1.246z"
                  />
                </svg>
              )}
              
              <span className="text-sm text-gray-500 truncate">{handle}</span>
              <span className="text-gray-500 hidden sm:inline">·</span>
              <span className="text-sm text-gray-500">{time}</span>

              {/* Sentiment Badge */}
              {sentiment && (
                <span 
                  className={`text-xs px-2 py-0.5 rounded-full font-medium border whitespace-nowrap ${getSentimentColor(sentiment.label)}`}
                >
                  {sentiment.label.toUpperCase()} • {Math.round(sentiment.confidence * 100)}%
                </span>
              )}

              {/* Flag Badge */}
              {flagLabel && (
                <span 
                  className={`text-xs px-2 py-0.5 rounded-full font-medium border whitespace-nowrap ${getFlagColor(flagLabel)}`}
                >
                  ⚠️ {flagLabel}
                </span>
              )}
            </div>

            {/* Content */}
            <div className="text-sm sm:text-base text-gray-900 mb-3 break-words whitespace-pre-wrap leading-normal">
              {parseMarkdown(content)}
            </div>

            {/* Images */}
            {images && images.length > 0 && (
              <div 
                className={`mb-3 rounded-2xl overflow-hidden cursor-pointer ${
                  images.length === 1 ? 'max-h-96' : 'grid grid-cols-2 gap-1 max-h-80'
                }`}
                onClick={() => {
                  setCurrentImageIndex(0);
                  setImageModalOpen(true);
                }}
              >
                {images.length === 1 ? (
                  <img 
                    src={images[0]} 
                    alt="Post" 
                    className="w-full h-full object-cover rounded-2xl"
                  />
                ) : (
                  images.slice(0, 4).map((img, idx) => (
                    <div key={idx} className="relative">
                      <img 
                        src={img} 
                        alt={`Post ${idx + 1}`} 
                        className="w-full h-40 object-cover rounded-lg"
                      />
                      {idx === 3 && images.length > 4 && (
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center rounded-lg">
                          <span className="text-white text-2xl font-bold">
                            +{images.length - 4}
                          </span>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center justify-between text-gray-500 max-w-md">
              <button 
                onClick={onReply}
                className="group flex items-center gap-1.5 sm:gap-2 hover:text-blue-600 transition-colors"
              >
                <div className="p-1.5 sm:p-2 rounded-full group-hover:bg-blue-50 transition-colors">
                  <MessageCircle className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={2} />
                </div>
                <span className="text-xs sm:text-sm">{replies.length}</span>
              </button>

              <button 
                onClick={onRetweet}
                className={`group flex items-center gap-1.5 sm:gap-2 transition-colors ${
                  isRetweeted ? 'text-green-600' : 'hover:text-green-600'
                }`}
              >
                <div className={`p-1.5 sm:p-2 rounded-full transition-colors ${
                  isRetweeted ? 'bg-green-50' : 'group-hover:bg-green-50'
                }`}>
                  <Repeat2 className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={2} />
                </div>
                <span className="text-xs sm:text-sm">{retweets}</span>
              </button>

              <button 
                onClick={onLike}
                className={`group flex items-center gap-1.5 sm:gap-2 transition-colors ${
                  isLiked ? 'text-pink-600' : 'hover:text-pink-600'
                }`}
              >
                <div className={`p-1.5 sm:p-2 rounded-full transition-colors ${
                  isLiked ? 'bg-pink-50' : 'group-hover:bg-pink-50'
                }`}>
                  <Heart 
                    className={`w-4 h-4 sm:w-5 sm:h-5 ${isLiked ? 'fill-current' : ''}`} 
                    strokeWidth={2} 
                  />
                </div>
                <span className="text-xs sm:text-sm">{likes}</span>
              </button>

              <div className="flex items-center gap-1.5 sm:gap-2">
                <BarChart className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={2} />
                <span className="text-xs sm:text-sm">{views}</span>
              </div>

              <button 
                onClick={onBookmark}
                className={`group p-1.5 sm:p-2 rounded-full transition-colors ${
                  isBookmarked ? 'text-blue-600 bg-blue-50' : 'hover:bg-gray-100'
                }`}
              >
                <Bookmark 
                  className={`w-4 h-4 sm:w-5 sm:h-5 ${isBookmarked ? 'fill-current' : ''}`} 
                  strokeWidth={2} 
                />
              </button>

              <button className="group p-1.5 sm:p-2 hover:bg-gray-100 rounded-full transition-colors">
                <Share className="w-4 h-4 sm:w-5 sm:h-5" strokeWidth={2} />
              </button>
            </div>

            {/* Toggle Replies */}
            {replies.length > 0 && (
              <button
                onClick={onToggleReplies}
                className="mt-3 flex items-center gap-2 text-blue-600 hover:text-blue-700 text-sm font-medium"
              >
                {showReplies ? (
                  <>
                    <ChevronUp className="w-4 h-4" />
                    Hide replies
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-4 h-4" />
                    Show {replies.length} {replies.length === 1 ? 'reply' : 'replies'}
                  </>
                )}
              </button>
            )}

            {/* Replies */}
            {showReplies && replies.length > 0 && (
              <div className="mt-4 space-y-3">
                {replies.map((reply) => (
                  <div 
                    key={reply.id} 
                    className="bg-gray-50 rounded-xl p-3 border border-gray-200 hover:border-gray-300 transition-colors"
                  >
                    <div className="flex gap-2 sm:gap-3">
                      <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                        <User className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-600" strokeWidth={2} />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 sm:gap-2 mb-1 flex-wrap">
                          <span className="font-bold text-xs sm:text-sm text-gray-900 truncate">
                            {reply.username}
                          </span>
                          
                          {reply.verified && (
                            <svg width="14" height="14" viewBox="0 0 22 22" className="text-blue-500 flex-shrink-0">
                              <path
                                fill="currentColor"
                                d="M20.396 11c-.018-.646-.215-1.275-.57-1.816-.354-.54-.852-.972-1.438-1.246.223-.607.27-1.264.14-1.897-.131-.634-.437-1.218-.882-1.687-.47-.445-1.053-.75-1.687-.882-.633-.13-1.29-.083-1.897.14-.273-.587-.704-1.086-1.245-1.44S11.647 1.62 11 1.604c-.646.017-1.273.213-1.813.568s-.969.854-1.24 1.44c-.608-.223-1.267-.272-1.902-.14-.635.13-1.22.436-1.69.882-.445.47-.749 1.055-.878 1.688-.13.633-.08 1.29.144 1.896-.587.274-1.087.705-1.443 1.245-.356.54-.555 1.17-.574 1.817.02.647.218 1.276.574 1.817.356.54.856.972 1.443 1.245-.224.606-.274 1.263-.144 1.896.13.634.433 1.218.877 1.688.47.443 1.054.747 1.687.878.633.132 1.29.084 1.897-.136.274.586.705 1.084 1.246 1.439.54.354 1.17.551 1.816.569.647-.016 1.276-.213 1.817-.567s.972-.854 1.245-1.44c.604.239 1.266.296 1.903.164.636-.132 1.22-.447 1.68-.907.46-.46.776-1.044.908-1.681s.075-1.299-.165-1.903c.586-.274 1.084-.705 1.439-1.246.354-.54.551-1.17.569-1.816zM9.662 14.85l-3.429-3.428 1.293-1.302 2.072 2.072 4.4-4.794 1.347 1.246z"
                              />
                            </svg>
                          )}
                          
                          <span className="text-xs sm:text-sm text-gray-500 truncate">{reply.handle}</span>
                          <span className="text-gray-500 hidden sm:inline">·</span>
                          <span className="text-xs sm:text-sm text-gray-500">{reply.time}</span>

                          {reply.sentiment && (
                            <span 
                              className={`text-9px sm:text-10px px-1.5 py-0.5 rounded-full font-medium whitespace-nowrap ml-auto ${getSentimentColor(reply.sentiment.label)}`}
                            >
                              {reply.sentiment.label.charAt(0).toUpperCase() + reply.sentiment.label.slice(1)} • {Math.round(reply.sentiment.confidence * 100)}%
                            </span>
                          )}
                        </div>

                        <div className="text-xs sm:text-sm text-gray-900 mb-2 break-words whitespace-pre-wrap leading-relaxed">
                          {parseMarkdown(reply.content)}
                        </div>

                        {reply.images && reply.images.length > 0 && (
                          <div className="mb-2 rounded-lg overflow-hidden">
                            <img 
                              src={reply.images[0]} 
                              alt="" 
                              className="w-full max-h-64 object-cover rounded-lg"
                            />
                          </div>
                        )}

                        <div className="flex items-center gap-2 text-gray-500">
                          <button 
                            onClick={() => onReplyLike?.(reply.id)}
                            className={`group flex items-center gap-1 transition-colors ${
                              reply.isLiked ? 'text-pink-600 hover:text-pink-600' : ''
                            }`}
                          >
                            <div className={`p-1.5 rounded-full transition-colors ${
                              reply.isLiked ? 'bg-pink-50' : 'group-hover:bg-pink-50'
                            }`}>
                              <Heart 
                                className={`w-3.5 h-3.5 ${reply.isLiked ? 'fill-current' : ''}`} 
                                strokeWidth={2} 
                              />
                            </div>
                            <span className="text-xs">{reply.likes}</span>
                          </button>

                          <div className="flex items-center gap-1">
                            <BarChart className="w-3.5 h-3.5" strokeWidth={2} />
                            <span className="text-xs">{reply.views}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Image Modal */}
      {imageModalOpen && images && (
        <div 
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-2 sm:p-4"
          onClick={() => setImageModalOpen(false)}
        >
          <button
            onClick={() => setImageModalOpen(false)}
            className="absolute top-2 right-2 sm:top-4 sm:right-4 text-white hover:bg-white/20 rounded-full p-1.5 sm:p-2 transition-colors z-10"
          >
            <X className="w-5 h-5 sm:w-6 sm:h-6" />
          </button>

          <div 
            className="relative max-w-4xl max-h-[90vh] w-full"
            onClick={(e) => e.stopPropagation()}
          >
            <img 
              src={images[currentImageIndex]} 
              alt="" 
              className="w-full h-full object-contain rounded-lg"
            />

            {images.length > 1 && (
              <>
                <button
                  onClick={prevImage}
                  className="absolute left-2 sm:left-4 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full p-2 sm:p-3 transition-colors"
                >
                  <ChevronLeft className="w-5 h-5 sm:w-6 sm:h-6" />
                </button>

                <button
                  onClick={nextImage}
                  className="absolute right-2 sm:right-4 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full p-2 sm:p-3 transition-colors"
                >
                  <ChevronRight className="w-5 h-5 sm:w-6 sm:h-6" />
                </button>
              </>
            )}

            <div className="absolute bottom-2 sm:bottom-4 left-1/2 -translate-x-1/2 bg-black/60 text-white text-xs sm:text-sm px-3 sm:px-4 py-1.5 sm:py-2 rounded-full">
              {currentImageIndex + 1} / {images.length}
            </div>
          </div>
        </div>
      )}
    </article>
  );
};

export default Post;
