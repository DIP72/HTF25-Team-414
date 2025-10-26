import { useState, useRef } from "react";
import { Image as ImageIcon, Smile, User, X, Sparkles, Loader2, ShieldAlert, Bold, Italic, Underline, Eye } from "lucide-react";
import aiService from "@/services/aiService";
import { parseMarkdown } from "@/utils/markdown";

interface CreateThreadProps {
  onPost: (content: string, images?: string[], labels?: string[], sentiment?: { label: string; confidence: number }) => void;
  currentUser: {
    username: string;
    handle: string;
    verified: boolean;
  };
}

const MAX_POST_CHARS = 1000;

const CreateThread = ({ onPost, currentUser }: CreateThreadProps) => {
  const [content, setContent] = useState("");
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pill, setPill] = useState<{ kind: "flagged" | "blocked"; label: string } | null>(null);
  const [showAIMenu, setShowAIMenu] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const compact = (t: string) => t.replace(/\s+/g, " ").trim();
  const remaining = MAX_POST_CHARS - content.length;
  const overLimit = remaining < 0;

  const handlePost = async () => {
    const base = compact(content);
    if (!base) return;

    if (base.length > MAX_POST_CHARS) {
      setPill({ kind: "flagged", label: `Over ${MAX_POST_CHARS}` });
      return;
    }

    setIsAnalyzing(true);
    setPill(null);

    try {
      const analysis = await aiService.analyzePost(base);
      const { moderation, sentiment, review_flag } = analysis;

      if (moderation.verdict === "blocked") {
        setPill({ kind: "blocked", label: moderation.labels?.[0] || "Blocked" });
        setIsAnalyzing(false);
        return;
      }

      if (moderation.verdict === "flagged" || review_flag) {
        setPill({ kind: "flagged", label: moderation.labels?.[0] || "Flagged" });
      }

      onPost(
        base,
        selectedImages.length > 0 ? selectedImages : undefined,
        moderation.verdict === "flagged" || review_flag ? moderation.labels : [],
        sentiment
      );

      setContent("");
      setSelectedImages([]);
      setPill(null);
    } catch (e) {
      onPost(base, selectedImages.length > 0 ? selectedImages : undefined);
      setContent("");
      setSelectedImages([]);
    } finally {
      setIsAnalyzing(false);
    }
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

      reader.onerror = () => {
        loadedCount++;
      };

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

  const handleClickOutside = () => {
    if (showAIMenu) setShowAIMenu(false);
  };

  return (
    <div className="bg-gray-50 rounded-2xl p-3 sm:p-4 mb-4" onClick={handleClickOutside}>
      <div className="flex gap-2 sm:gap-3">
        <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
          <User className="w-4 h-4 sm:w-5 sm:h-5 text-gray-600" strokeWidth={2} />
        </div>

        <div className="flex-1">
          <textarea
            ref={textareaRef}
            placeholder="What's happening?"
            className={`w-full bg-transparent text-gray-900 text-base sm:text-[18px] placeholder:text-gray-500 border-none outline-none resize-none min-h-[80px] sm:min-h-[100px] mb-2 ${
              overLimit ? "ring-2 ring-red-300 rounded-lg p-2" : ""
            }`}
            value={content}
            onChange={(e) => {
              setContent(e.target.value);
              setPill(null);
            }}
            disabled={isAnalyzing || isProcessing}
          />

          <div className="flex items-center justify-between mb-3">
            <div className="text-xs text-gray-500">
              {overLimit ? (
                <span className="text-red-600 font-medium">Over limit by {Math.abs(remaining)} chars</span>
              ) : (
                <span className={remaining < 100 ? "text-amber-600" : ""}>{remaining} left</span>
              )}
            </div>
            {pill && (
              <span
                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium ${
                  pill.kind === "blocked" ? "bg-red-100 text-red-700" : "bg-amber-100 text-amber-700"
                }`}
                title={pill.label}
              >
                <ShieldAlert className="w-3.5 h-3.5" />
                {pill.label}
              </span>
            )}
          </div>

          {selectedImages.length > 0 && (
            <div className={`mb-3 grid gap-2 ${
              selectedImages.length === 1 ? "grid-cols-1" : 
              selectedImages.length === 2 ? "grid-cols-2" :
              selectedImages.length === 3 ? "grid-cols-3" :
              "grid-cols-2"
            }`}>
              {selectedImages.map((img, idx) => (
                <div key={idx} className="relative rounded-xl overflow-hidden">
                  <img 
                    src={img} 
                    alt={`Preview ${idx + 1}`} 
                    className="w-full h-48 object-cover rounded-xl" 
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23ddd' width='100' height='100'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23999'%3EError%3C/text%3E%3C/svg%3E";
                    }}
                  />
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
            <div className="mb-3 p-3 bg-white rounded-xl border border-gray-200">
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
            <div className="flex items-center gap-0.5 sm:gap-1">
              <button
                onClick={() => applyFormat("bold")}
                className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Bold (select text first)"
                disabled={isAnalyzing || isProcessing}
              >
                <Bold className="w-4 h-4" strokeWidth={2.5} />
              </button>

              <button
                onClick={() => applyFormat("italic")}
                className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Italic (select text first)"
                disabled={isAnalyzing || isProcessing}
              >
                <Italic className="w-4 h-4" strokeWidth={2.5} />
              </button>

              <button
                onClick={() => applyFormat("underline")}
                className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Underline (select text first)"
                disabled={isAnalyzing || isProcessing}
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
                disabled={isAnalyzing || isProcessing}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40"
                title="Upload images (multiple)"
                disabled={isAnalyzing || isProcessing}
              >
                <ImageIcon className="w-5 h-5" strokeWidth={2} />
              </button>

              <button
                onClick={() => setShowPreview(!showPreview)}
                className={`p-2 rounded-lg transition-all ${
                  showPreview ? "bg-blue-100 text-blue-600" : "text-gray-600 hover:bg-gray-200 hover:text-gray-900"
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
                  disabled={!content.trim() || isAnalyzing || isProcessing}
                  className="p-2 text-blue-600 hover:bg-blue-50 hover:text-blue-700 rounded-full transition-all disabled:opacity-40 disabled:cursor-not-allowed"
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

              <button
                className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all disabled:opacity-40"
                disabled={isAnalyzing || isProcessing}
                title="Emoji"
              >
                <Smile className="w-5 h-5" strokeWidth={2} />
              </button>
            </div>

            <button
              onClick={handlePost}
              disabled={!content.trim() || isAnalyzing || isProcessing || overLimit}
              className={`px-5 py-2 rounded-full text-white text-[15px] font-bold transition-all ${
                !content.trim() || isAnalyzing || isProcessing || overLimit
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-black hover:bg-gray-800"
              }`}
            >
              {isAnalyzing ? "Analyzing..." : "Post"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreateThread;
