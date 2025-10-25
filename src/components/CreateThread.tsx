// src/components/CreateThread.tsx
import { useState } from "react";
import { Image as ImageIcon, Smile, MapPin, BarChart3, Calendar, User, X } from "lucide-react";

interface CreateThreadProps {
  onPost: (content: string, image?: string) => void;
}

const CreateThread = ({ onPost }: CreateThreadProps) => {
  const [content, setContent] = useState("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const handlePost = () => {
    if (content.trim()) {
      onPost(content, selectedImage || undefined);
      setContent("");
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
    <div className="bg-gray-50 rounded-2xl p-4 mb-4">
      <div className="flex gap-3">
        <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
          <User className="w-6 h-6 text-gray-600" strokeWidth={2} />
        </div>

        <div className="flex-1">
          <textarea
            placeholder="What's happening?"
            className="w-full bg-transparent text-gray-900 text-[18px] placeholder:text-gray-500 border-none outline-none resize-none min-h-[80px] mb-3"
            value={content}
            onChange={(e) => setContent(e.target.value)}
          />

          {selectedImage && (
            <div className="relative mb-3 rounded-2xl overflow-hidden">
              <img src={selectedImage} alt="Preview" className="w-full max-h-80 object-cover rounded-xl" />
              <button
                onClick={() => setSelectedImage(null)}
                className="absolute top-2 right-2 w-8 h-8 bg-black/70 hover:bg-black rounded-full flex items-center justify-center transition-colors"
              >
                <X className="w-4 h-4 text-white" />
              </button>
            </div>
          )}

          <div className="flex items-center justify-between pt-3 border-t border-gray-200">
            <div className="flex items-center gap-1">
              <label className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all cursor-pointer">
                <ImageIcon className="w-5 h-5" strokeWidth={2} />
                <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
              </label>
              <button className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all">
                <BarChart3 className="w-5 h-5" strokeWidth={2} />
              </button>
              <button className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all">
                <Smile className="w-5 h-5" strokeWidth={2} />
              </button>
              <button className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all">
                <Calendar className="w-5 h-5" strokeWidth={2} />
              </button>
              <button className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all">
                <MapPin className="w-5 h-5" strokeWidth={2} />
              </button>
            </div>

            <button
              onClick={handlePost}
              disabled={content.trim().length === 0}
              className={`px-5 py-2 rounded-full text-white text-[15px] font-bold transition-all ${
                content.trim().length === 0
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-black hover:bg-gray-800"
              }`}
            >
              Post
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreateThread;
