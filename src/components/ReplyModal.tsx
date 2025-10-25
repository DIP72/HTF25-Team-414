// src/components/ReplyModal.tsx
import { useState } from "react";
import { X, User, Sparkles, Image as ImageIcon } from "lucide-react";

interface ReplyModalProps {
  post: {
    username: string;
    handle: string;
    content: string;
  };
  onClose: () => void;
  onSubmit: (content: string, image?: string) => void;
}

const ReplyModal = ({ post, onClose, onSubmit }: ReplyModalProps) => {
  const [replyContent, setReplyContent] = useState("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const handleSubmit = () => {
    if (replyContent.trim()) {
      onSubmit(replyContent, selectedImage || undefined);
      setReplyContent("");
      setSelectedImage(null);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl max-w-[600px] w-full max-h-[80vh] overflow-y-auto shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-gray-200 sticky top-0 bg-white">
          <h2 className="text-lg font-bold text-gray-900">Reply</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-gray-600" />
          </button>
        </div>

        <div className="p-4 border-b border-gray-100 bg-gray-50">
          <div className="flex gap-3">
            <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0">
              <User className="w-5 h-5 text-gray-400" strokeWidth={2} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-bold text-sm text-gray-900">{post.username}</span>
                <span className="text-gray-500 text-sm">@{post.handle}</span>
              </div>
              <p className="text-sm text-gray-700">{post.content}</p>
            </div>
          </div>
        </div>

        <div className="p-4">
          <div className="flex gap-3">
            <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0">
              <User className="w-5 h-5 text-gray-400" strokeWidth={2} />
            </div>
            <div className="flex-1">
              <p className="text-sm text-gray-500 mb-2">
                Replying to <span className="text-blue-600">@{post.handle}</span>
              </p>
              <textarea
                placeholder="Post your reply"
                className="w-full bg-transparent text-gray-900 text-[15px] placeholder:text-gray-400 border-none outline-none resize-none min-h-[100px] mb-3"
                value={replyContent}
                onChange={(e) => setReplyContent(e.target.value)}
                autoFocus
              />
              {selectedImage && (
                <div className="relative mb-3">
                  <img src={selectedImage} alt="Preview" className="rounded-xl max-h-64 object-cover" />
                  <button
                    onClick={() => setSelectedImage(null)}
                    className="absolute top-2 right-2 w-8 h-8 bg-black/70 hover:bg-black rounded-full flex items-center justify-center"
                  >
                    <X className="w-4 h-4 text-white" />
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between p-4 border-t border-gray-200">
          <div className="flex items-center gap-2">
            <label className="p-2 text-blue-600 hover:bg-blue-50 rounded-full transition-all cursor-pointer">
              <ImageIcon className="w-5 h-5" strokeWidth={2} />
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </label>
            <button className="flex items-center gap-2 px-3 py-2 text-blue-600 hover:bg-blue-50 rounded-full transition-all">
              <Sparkles className="w-4 h-4" strokeWidth={2} />
              <span className="text-sm font-semibold">AI</span>
            </button>
          </div>

          <button
            onClick={handleSubmit}
            disabled={replyContent.trim().length === 0}
            className={`px-6 py-2 rounded-full text-white text-sm font-bold transition-all ${
              replyContent.trim().length === 0
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-blue-500 hover:bg-blue-600 active:scale-95"
            }`}
          >
            Reply
          </button>
        </div>
      </div>
    </div>
  );
};

export default ReplyModal;
