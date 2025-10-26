import { X, Image as ImageIcon, User, Sparkles, Loader2, Bold, Italic, Underline, Eye } from "lucide-react";
import { useState, useRef } from "react";
import aiService from "@/services/aiService";
import { parseMarkdown } from "@/utils/markdown";

interface ReplyModalProps {
  post: {
    username: string;
    handle: string;
    content: string;
  };
  currentUser: {
    username: string;
    handle: string;
    verified: boolean;
  };
  onClose: () => void;
  onSubmit: (content: string, images?: string[]) => void;
}

const MAX_POST_CHARS = 1000;

const ReplyModal = ({ post, currentUser, onClose, onSubmit }: ReplyModalProps) => {
  const [content, setContent] = useState("");
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showAIMenu, setShowAIMenu] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const remaining = MAX_POST_CHARS - content.length;
  const overLimit = remaining < 0;

  const handleSubmit = () => {
    if (!content.trim() || overLimit) return;
    onSubmit(content.trim(), selectedImages.length > 0 ? selectedImages : undefined);
    onClose();
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    const newImages: string[] = [];
    let loadedCount = 0;

    fileArray.forEach((file) => {
      if (!file.type.startsWith('image/')) {
        loadedCount++;
        return;
      }

      const reader = new FileReader();
      reader.onload = () => {
        newImages.push(reader.result as string);
        loadedCount++;
        if (loadedCount === fileArray.length) {
          setSelectedImages((prev) => [...prev, ...newImages]);
        }
      };
      reader.onerror = () => { loadedCount++; };
      reader.readAsDataURL(file);
    });

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const removeImage = (index: number) => {
    setSelectedImages((prev) => prev.filter((_, i) => i !== index));
  };

  const handleAIRewrite = async () => {
    if (!content.trim()) return;
    setIsProcessing(true);
    setShowAIMenu(false);

    try {
      const { draft } = await aiService.draftPost(content);
      const cleaned = draft.replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "").replace(/#\w+/g, "").trim();
      setContent(cleaned);
    } catch (e) {
      console.error("AI Rewrite failed:", e);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAICondense = async () => {
    if (!content.trim()) return;
    setIsProcessing(true);
    setShowAIMenu(false);

    try {
      const { draft } = await aiService.condenseToPost(content);
      const cleaned = draft.replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, "").replace(/#\w+/g, "").trim();
      setContent(cleaned);
    } catch (e) {
      console.error("AI Condense failed:", e);
    } finally {
      setIsProcessing(false);
    }
  };

  const applyFormat = (formatType: "bold" | "italic" | "underline") => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = content.substring(start, end);

    if (!selectedText) return;

    let formatted: string;
    switch (formatType) {
      case "bold":
        formatted = `**${selectedText}**`;
        break;
      case "italic":
        formatted = `*${selectedText}*`;
        break;
      case "underline":
        formatted = `__${selectedText}__`;
        break;
    }

    const newContent = content.substring(0, start) + formatted + content.substring(end);
    setContent(newContent);

    setTimeout(() => {
      textarea.focus();
      const newPos = start + formatted.length;
      textarea.setSelectionRange(newPos, newPos);
    }, 0);
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="sticky top-0 bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between rounded-t-2xl">
          <h2 className="text-lg font-bold text-gray-900">Reply</h2>
          <button onClick={onClose} className="text-gray-500 hover:bg-gray-100 rounded-full p-2 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4">
          <div className="mb-4 pb-4 border-b border-gray-200">
            <div className="flex gap-3">
              <div className="w-9 h-9 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                <User className="w-4 h-4 text-gray-600" strokeWidth={2} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-bold text-sm text-gray-900">{post.username}</span>
                  <span className="text-sm text-gray-500">@{post.handle}</span>
                </div>
                <p className="text-sm text-gray-900">{post.content}</p>
              </div>
            </div>
          </div>

          <div className="flex gap-3">
            <div className="w-9 h-9 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
              <User className="w-4 h-4 text-white" strokeWidth={2} />
            </div>
            <div className="flex-1">
              <textarea
                ref={textareaRef}
                placeholder="Write your reply..."
                className={`w-full bg-transparent text-gray-900 text-base placeholder:text-gray-500 border-none outline-none resize-none min-h-[100px] mb-2 ${
                  overLimit ? "ring-2 ring-red-300 rounded-lg p-2" : ""
                }`}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                disabled={isProcessing}
              />

              <div className="flex items-center justify-between mb-3">
                <div className="text-xs text-gray-500">
                  {overLimit ? (
                    <span className="text-red-600 font-medium">Over limit by {Math.abs(remaining)} chars</span>
                  ) : (
                    <span className={remaining < 100 ? "text-amber-600" : ""}>{remaining} left</span>
                  )}
                </div>
              </div>

              {selectedImages.length > 0 && (
                <div className={`mb-3 grid gap-2 ${selectedImages.length === 1 ? "grid-cols-1" : "grid-cols-2"}`}>
                  {selectedImages.map((img, idx) => (
                    <div key={idx} className="relative rounded-xl overflow-hidden">
                      <img src={img} alt={`Preview ${idx + 1}`} className="w-full h-48 object-cover rounded-xl" />
                      <button
                        onClick={() => removeImage(idx)}
                        className="absolute top-2 right-2 w-7 h-7 bg-black/70 hover:bg-black rounded-full flex items-center justify-center transition-colors"
                      >
                        <X className="w-4 h-4 text-white" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {showPreview && content.trim() && (
                <div className="mb-3 p-3 bg-gray-50 rounded-xl border border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-gray-500">Preview</span>
                    <button onClick={() => setShowPreview(false)} className="text-xs text-blue-600 hover:underline">
                      Hide
                    </button>
                  </div>
                  <div className="text-sm text-gray-900 leading-normal break-words whitespace-pre-wrap">
                    {parseMarkdown(content)}
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between pt-3 border-t border-gray-200">
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => applyFormat("bold")}
                    className="p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40"
                    title="Bold"
                    disabled={isProcessing}
                  >
                    <Bold className="w-4 h-4" strokeWidth={2.5} />
                  </button>

                  <button
                    onClick={() => applyFormat("italic")}
                    className="p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40"
                    title="Italic"
                    disabled={isProcessing}
                  >
                    <Italic className="w-4 h-4" strokeWidth={2.5} />
                  </button>

                  <button
                    onClick={() => applyFormat("underline")}
                    className="p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40"
                    title="Underline"
                    disabled={isProcessing}
                  >
                    <Underline className="w-4 h-4" strokeWidth={2.5} />
                  </button>

                  <div className="w-px h-6 bg-gray-300 mx-1" />

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    multiple
                    className="hidden"
                    onChange={handleImageUpload}
                    disabled={isProcessing}
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40"
                    title="Upload images"
                    disabled={isProcessing}
                  >
                    <ImageIcon className="w-5 h-5" strokeWidth={2} />
                  </button>

                  <button
                    onClick={() => setShowPreview(!showPreview)}
                    className={`p-2 rounded-lg transition-all ${
                      showPreview ? "bg-blue-100 text-blue-600" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                    }`}
                    title="Toggle preview"
                    disabled={!content.trim()}
                  >
                    <Eye className="w-5 h-5" strokeWidth={2} />
                  </button>

                  <div className="w-px h-6 bg-gray-300 mx-1" />

                  <div className="relative" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => setShowAIMenu(!showAIMenu)}
                      disabled={!content.trim() || isProcessing}
                      className="p-2 text-blue-600 hover:bg-blue-50 hover:text-blue-700 rounded-full transition-all disabled:opacity-40"
                      title="AI Tools"
                    >
                      {isProcessing ? (
                        <Loader2 className="w-5 h-5 animate-spin" strokeWidth={2} />
                      ) : (
                        <Sparkles className="w-5 h-5" strokeWidth={2} />
                      )}
                    </button>

                    {showAIMenu && (
                      <div className="absolute bottom-full left-0 mb-2 bg-white rounded-xl shadow-lg border border-gray-200 py-1.5 min-w-[180px] z-10">
                        <button
                          onClick={handleAIRewrite}
                          className="w-full px-4 py-2.5 text-left text-sm hover:bg-gray-100 transition-colors flex items-center gap-2.5 text-gray-700"
                        >
                          <Sparkles className="w-4 h-4 text-blue-600" />
                          Rewrite with AI
                        </button>
                        <button
                          onClick={handleAICondense}
                          className="w-full px-4 py-2.5 text-left text-sm hover:bg-gray-100 transition-colors flex items-center gap-2.5 text-gray-700"
                        >
                          <Sparkles className="w-4 h-4 text-blue-600" />
                          Make shorter
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={!content.trim() || isProcessing || overLimit}
                  className={`px-5 py-2 rounded-full text-white text-sm font-bold transition-all ${
                    !content.trim() || isProcessing || overLimit
                      ? "bg-gray-400 cursor-not-allowed"
                      : "bg-blue-600 hover:bg-blue-700"
                  }`}
                >
                  Reply
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReplyModal;
