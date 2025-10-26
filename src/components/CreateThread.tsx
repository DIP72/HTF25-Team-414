// src/components/CreateThread.tsx
import { useState, useRef } from "react";
import { Image as ImageIcon, Smile, User, X, Sparkles, Loader2, ShieldAlert, Bold, Italic, Underline } from "lucide-react";
import aiService from "@/services/aiService";

interface CreateThreadProps {
  onPost: (
    content: string,
    image?: string,
    labels?: string[],
    sentiment?: { label: string; confidence: number }
  ) => void;
}

const MAX_POST_CHARS = 1000;

const CreateThread = ({ onPost }: CreateThreadProps) => {
  const [content, setContent] = useState("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pill, setPill] = useState<{ kind: "flagged" | "blocked"; label: string } | null>(null);
  const [showAIMenu, setShowAIMenu] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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

      onPost(base, selectedImage ?? undefined, moderation.verdict === "flagged" || review_flag ? moderation.labels : [], sentiment);
      setContent("");
      setSelectedImage(null);
      setPill(null);
    } catch (e) {
      onPost(base, selectedImage ?? undefined);
      setContent("");
      setSelectedImage(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onloadend = () => setSelectedImage(reader.result as string);
    reader.readAsDataURL(f);
  };

  // AI Features - Trust the backend prompts, minimal cleaning
  const handleAIRewrite = async () => {
    if (!content.trim()) return;
    setIsProcessing(true);
    setShowAIMenu(false);
    try {
      const { draft } = await aiService.draftPost(content);
      // Only remove emojis/hashtags if they somehow slip through
      const cleaned = draft.replace(/[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu, '').replace(/#\w+/g, '').trim();
      setContent(cleaned);
    } catch (e) {
      console.error(e);
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
      // Only remove emojis/hashtags if they somehow slip through
      const cleaned = draft.replace(/[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu, '').replace(/#\w+/g, '').trim();
      setContent(cleaned);
    } catch (e) {
      console.error(e);
    } finally {
      setIsProcessing(false);
    }
  };

  // Text formatting
  const applyFormat = (formatType: 'bold' | 'italic' | 'underline') => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = content.substring(start, end);
    
    if (!selectedText) return;

    let formatted = "";
    switch (formatType) {
      case 'bold':
        formatted = `**${selectedText}**`;
        break;
      case 'italic':
        formatted = `*${selectedText}*`;
        break;
      case 'underline':
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
        <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 sm:w-6 sm:h-6 text-gray-600" strokeWidth={2} />
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

          {/* Single line with all tools */}
          <div className="flex items-center justify-between pt-3 border-t border-gray-200">
            <div className="flex items-center gap-0.5 sm:gap-1">
              {/* Formatting tools */}
              <button
                onClick={() => applyFormat('bold')}
                className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Bold (select text first)"
                disabled={isAnalyzing || isProcessing}
              >
                <Bold className="w-4 h-4" strokeWidth={2.5} />
              </button>
              <button
                onClick={() => applyFormat('italic')}
                className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Italic (select text first)"
                disabled={isAnalyzing || isProcessing}
              >
                <Italic className="w-4 h-4" strokeWidth={2.5} />
              </button>
              <button
                onClick={() => applyFormat('underline')}
                className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Underline (select text first)"
                disabled={isAnalyzing || isProcessing}
              >
                <Underline className="w-4 h-4" strokeWidth={2.5} />
              </button>

              <div className="w-px h-6 bg-gray-300 mx-1" />

              {/* Media tools */}
              <label className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all cursor-pointer disabled:opacity-40">
                <ImageIcon className="w-5 h-5" strokeWidth={2} />
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleImageUpload} 
                  className="hidden" 
                  disabled={isAnalyzing || isProcessing} 
                />
              </label>

              {/* AI Menu */}
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
