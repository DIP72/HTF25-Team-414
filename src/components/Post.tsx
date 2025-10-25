// src/components/Post.tsx
import { Heart, MessageCircle, Repeat2, Share, MoreHorizontal, User, BarChart, X, ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";

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
  aiModerated?: boolean;
  aiSummary?: string;
  isLiked?: boolean;
  isRetweeted?: boolean;
  showReplies?: boolean;
  onLike?: () => void;
  onReply?: () => void;
  onRetweet?: () => void;
  onToggleReplies?: () => void;
}

const Post = ({ 
  username,
  handle,
  verified, 
  time, 
  content, 
  images,
  replies,
  retweets,
  likes,
  views,
  isLiked = false,
  isRetweeted = false,
  showReplies = false,
  onLike,
  onReply,
  onRetweet,
  onToggleReplies
}: PostProps) => {
  const [imageModalOpen, setImageModalOpen] = useState(false);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const openImageModal = (index: number) => {
    setCurrentImageIndex(index);
    setImageModalOpen(true);
  };

  const nextImage = () => {
    if (images) {
      setCurrentImageIndex((prev) => (prev + 1) % images.length);
    }
  };

  const prevImage = () => {
    if (images) {
      setCurrentImageIndex((prev) => (prev - 1 + images.length) % images.length);
    }
  };

  return (
    <div className="mb-4">
      {/* Post Card */}
      <article className="bg-gray-50 rounded-2xl p-4 hover:bg-gray-100 transition-colors duration-200">
        <div className="flex gap-3">
          <div className="relative flex-shrink-0">
            <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center">
              <User className="w-6 h-6 text-gray-600" strokeWidth={2} />
            </div>
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-bold text-[15px] text-gray-900 hover:underline cursor-pointer">{username}</span>
              {verified && (
                <svg width="18" height="18" viewBox="0 0 22 22" className="text-blue-500">
                  <path fill="currentColor" d="M20.396 11c-.018-.646-.215-1.275-.57-1.816-.354-.54-.852-.972-1.438-1.246.223-.607.27-1.264.14-1.897-.131-.634-.437-1.218-.882-1.687-.47-.445-1.053-.75-1.687-.882-.633-.13-1.29-.083-1.897.14-.273-.587-.704-1.086-1.245-1.44S11.647 1.62 11 1.604c-.646.017-1.273.213-1.813.568s-.969.854-1.24 1.44c-.608-.223-1.267-.272-1.902-.14-.635.13-1.22.436-1.69.882-.445.47-.749 1.055-.878 1.688-.13.633-.08 1.29.144 1.896-.587.274-1.087.705-1.443 1.245-.356.54-.555 1.17-.574 1.817.02.647.218 1.276.574 1.817.356.54.856.972 1.443 1.245-.224.606-.274 1.263-.144 1.896.13.634.433 1.218.877 1.688.47.443 1.054.747 1.687.878.633.132 1.29.084 1.897-.136.274.586.705 1.084 1.246 1.439.54.354 1.17.551 1.816.569.647-.016 1.276-.213 1.817-.567s.972-.854 1.245-1.44c.604.239 1.266.296 1.903.164.636-.132 1.22-.447 1.68-.907.46-.46.776-1.044.908-1.681s.075-1.299-.165-1.903c.586-.274 1.084-.705 1.439-1.246.354-.54.551-1.17.569-1.816zM9.662 14.85l-3.429-3.428 1.293-1.302 2.072 2.072 4.4-4.794 1.347 1.246z"/>
                </svg>
              )}
              <span className="text-[15px] text-gray-500">@{handle}</span>
              <span className="text-gray-500">·</span>
              <span className="text-[15px] text-gray-500">{time}</span>
              <button className="ml-auto text-gray-500 hover:bg-gray-200 hover:text-gray-900 rounded-full p-2 transition-all duration-200">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>

            <p className="text-[15px] text-gray-900 mb-3 leading-normal whitespace-pre-line">
              {content}
            </p>

            {/* Images with Click to View */}
            {images && images.length > 0 && (
              <div className={`mb-3 rounded-2xl overflow-hidden cursor-pointer ${
                images.length === 1 ? "grid grid-cols-1" : 
                images.length === 2 ? "grid grid-cols-2 gap-2" : 
                "grid grid-cols-2 gap-2"
              }`}>
                {images.map((img, idx) => (
                  <div 
                    key={idx} 
                    className="relative aspect-video overflow-hidden bg-gray-200 rounded-xl hover:opacity-90 transition-opacity"
                    onClick={() => openImageModal(idx)}
                  >
                    <img 
                      src={img} 
                      alt="" 
                      className="w-full h-full object-cover"
                    />
                    {images.length > 1 && (
                      <div className="absolute top-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded-full">
                        {idx + 1}/{images.length}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="flex items-center justify-between max-w-[425px] mt-3">
              <button
                onClick={onReply}
                className="group flex items-center gap-1 text-gray-500 hover:text-gray-900 transition-colors duration-200"
              >
                <div className="p-2 rounded-full group-hover:bg-gray-200 transition-colors duration-200">
                  <MessageCircle className="w-5 h-5" strokeWidth={2} />
                </div>
                <span className="text-sm">{replies.length}</span>
              </button>

              <button 
                onClick={onRetweet}
                className={`group flex items-center gap-1 transition-colors duration-200 ${
                  isRetweeted ? "text-green-600" : "text-gray-500 hover:text-gray-900"
                }`}
              >
                <div className={`p-2 rounded-full transition-colors duration-200 ${
                  isRetweeted ? "bg-green-50" : "group-hover:bg-gray-200"
                }`}>
                  <Repeat2 className="w-5 h-5" strokeWidth={2} />
                </div>
                <span className="text-sm">{retweets}</span>
              </button>

              <button
                onClick={onLike}
                className={`group flex items-center gap-1 transition-colors duration-200 ${
                  isLiked ? "text-pink-600" : "text-gray-500 hover:text-gray-900"
                }`}
              >
                <div className={`p-2 rounded-full transition-colors duration-200 ${
                  isLiked ? "bg-pink-50" : "group-hover:bg-gray-200"
                }`}>
                  <Heart 
                    className={`w-5 h-5 ${isLiked ? "fill-current" : ""}`}
                    strokeWidth={2}
                  />
                </div>
                <span className="text-sm">{likes}</span>
              </button>

              <button className="group flex items-center gap-1 text-gray-500 hover:text-gray-900 transition-colors duration-200">
                <div className="p-2 rounded-full group-hover:bg-gray-200 transition-colors duration-200">
                  <BarChart className="w-5 h-5" strokeWidth={2} />
                </div>
                <span className="text-sm">{views}</span>
              </button>

              <button className="group text-gray-500 hover:text-gray-900 transition-colors duration-200">
                <div className="p-2 rounded-full group-hover:bg-gray-200 transition-colors duration-200">
                  <Share className="w-5 h-5" strokeWidth={2} />
                </div>
              </button>
            </div>

            {/* Inline Replies */}
            {replies.length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <button
                  onClick={onToggleReplies}
                  className="text-sm text-gray-900 hover:underline font-medium mb-3"
                >
                  {showReplies ? `Hide ${replies.length} ${replies.length === 1 ? 'reply' : 'replies'}` : `View ${replies.length} ${replies.length === 1 ? 'reply' : 'replies'}`}
                </button>

                {showReplies && (
                  <div className="space-y-3 mt-3">
                    {replies.map((reply) => (
                      <div key={reply.id} className="flex gap-3 pl-3 border-l-2 border-gray-300">
                        <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                          <User className="w-4 h-4 text-gray-600" strokeWidth={2} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-bold text-sm text-gray-900">{reply.username}</span>
                            <span className="text-sm text-gray-500">@{reply.handle}</span>
                            <span className="text-gray-500">·</span>
                            <span className="text-sm text-gray-500">{reply.time}</span>
                          </div>
                          <p className="text-sm text-gray-900 mb-2">{reply.content}</p>
                          <div className="flex items-center gap-4 text-gray-500">
                            <button className="flex items-center gap-1 hover:text-pink-600 transition-colors">
                              <Heart className="w-4 h-4" strokeWidth={2} />
                              <span className="text-xs">{reply.likes}</span>
                            </button>
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

      {/* Image Modal */}
      {imageModalOpen && images && (
        <div 
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4"
          onClick={() => setImageModalOpen(false)}
        >
          <button
            onClick={() => setImageModalOpen(false)}
            className="absolute top-4 right-4 text-white hover:bg-white/20 rounded-full p-2 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>

          <div className="relative max-w-4xl max-h-[90vh] w-full" onClick={(e) => e.stopPropagation()}>
            <img 
              src={images[currentImageIndex]} 
              alt="" 
              className="w-full h-full object-contain rounded-lg"
            />

            {images.length > 1 && (
              <>
                <button
                  onClick={prevImage}
                  className="absolute left-4 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full p-3 transition-colors"
                >
                  <ChevronLeft className="w-6 h-6" />
                </button>
                <button
                  onClick={nextImage}
                  className="absolute right-4 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full p-3 transition-colors"
                >
                  <ChevronRight className="w-6 h-6" />
                </button>
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/60 text-white text-sm px-4 py-2 rounded-full">
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
