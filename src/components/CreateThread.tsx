// src/components/CreateThread.tsx
import { useState } from "react";
import { Image as ImageIcon, Smile, User, X, AlertCircle, Sparkles, Loader2 } from "lucide-react";
import { aiService } from "@/services/aiService";

interface CreateThreadProps {
  onPost: (content: string, image?: string, labels?: string[]) => void;
}

const CreateThread = ({ onPost }: CreateThreadProps) => {
  const [content, setContent] = useState("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDrafting, setIsDrafting] = useState(false);
  const [moderationLabels, setModerationLabels] = useState<string[]>([]);
  const [moderationWarning, setModerationWarning] = useState<string | null>(null);
  const [draftPrompt, setDraftPrompt] = useState("");
  const [showDraftInput, setShowDraftInput] = useState(false);

  const handleDraft = async () => {
    if (!draftPrompt.trim()) return;

    setIsDrafting(true);
    try {
      const result = await aiService.draftPost(draftPrompt);
      setContent(result.draft);
      setShowDraftInput(false);
      setDraftPrompt("");
    } catch (error) {
      console.error("Drafting failed:", error);
    } finally {
      setIsDrafting(false);
    }
  };

  const handlePost = async () => {
    if (!content.trim()) return;

    setIsAnalyzing(true);
    setModerationWarning(null);

    try {
      const analysis = await aiService.analyzePost(content);

      if (analysis.moderation.verdict === "blocked") {
        setModerationWarning(
          `⛔ This post cannot be published. ${analysis.moderation.reason}`
        );
        setModerationLabels(analysis.moderation.labels);
        setIsAnalyzing(false);
        return;
      }

      const labels = analysis.moderation.verdict === "flagged" 
        ? analysis.moderation.labels 
        : [];

      if (labels.length > 0) {
        setModerationWarning(
          `⚠️ This post will be flagged with: ${labels.join(", ")}`
        );
      }

      // Post with labels
      onPost(content, selectedImage || undefined, labels);
      setContent("");
      setSelectedImage(null);
      setModerationWarning(null);
      setModerationLabels([]);
      
    } catch (error) {
      console.error("AI analysis failed:", error);
      onPost(content, selectedImage || undefined);
      setContent("");
      setSelectedImage(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setSelectedImage(reader.result as string);
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
          {/* AI Draft Input */}
          {showDraftInput && (
            <div className="mb-3 p-3 bg-blue-50 rounded-xl border border-blue-200">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-semibold text-blue-900">AI Draft Assistant</span>
              </div>
              <input
                type="text"
                placeholder="What do you want to post about?"
                className="w-full bg-white rounded-lg px-3 py-2 text-sm border border-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500 mb-2"
                value={draftPrompt}
                onChange={(e) => setDraftPrompt(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleDraft()}
              />
              <div className="flex gap-2">
                <button
                  onClick={handleDraft}
                  disabled={!draftPrompt.trim() || isDrafting}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg disabled:bg-gray-400 flex items-center gap-1"
                >
                  {isDrafting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3" />}
                  {isDrafting ? "Drafting..." : "Generate"}
                </button>
                <button
                  onClick={() => setShowDraftInput(false)}
                  className="px-3 py-1 bg-gray-200 hover:bg-gray-300 text-gray-700 text-sm rounded-lg"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          <textarea
            placeholder="What's happening?"
            className="w-full bg-transparent text-gray-900 text-[18px] placeholder:text-gray-500 border-none outline-none resize-none min-h-[80px] mb-3"
            value={content}
            onChange={(e) => {
              setContent(e.target.value);
              setModerationWarning(null);
              setModerationLabels([]);
            }}
            disabled={isAnalyzing || isDrafting}
          />

          {/* Moderation Warning with Labels */}
          {moderationWarning && (
            <div className="mb-3 p-3 rounded-xl bg-yellow-50 border border-yellow-200">
              <div className="flex items-start gap-2 mb-2">
                <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-yellow-800">{moderationWarning}</p>
              </div>
              {moderationLabels.length > 0 && (
                <div className="flex gap-2 flex-wrap mt-2">
                  {moderationLabels.map((label, idx) => (
                    <span key={idx} className="px-2 py-1 bg-red-100 text-red-700 text-xs font-semibold rounded-full">
                      {label}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {selectedImage && (
            <div className="relative mb-3 rounded-2xl overflow-hidden">
              <img src={selectedImage} alt="Preview" className="w-full max-h-80 object-cover rounded-xl" />
              <button
                onClick={() => setSelectedImage(null)}
                className="absolute top-2 right-2 w-8 h-8 bg-black/70 hover:bg-black rounded-full flex items-center justify-center"
              >
                <X className="w-4 h-4 text-white" />
              </button>
            </div>
          )}

          <div className="flex items-center justify-between pt-3 border-t border-gray-200">
            <div className="flex items-center gap-1">
              <label className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all cursor-pointer">
                <ImageIcon className="w-5 h-5" strokeWidth={2} />
                <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" disabled={isAnalyzing} />
              </label>
              <button 
                onClick={() => setShowDraftInput(!showDraftInput)}
                className="p-2 text-blue-600 hover:bg-blue-50 hover:text-blue-700 rounded-full transition-all"
                disabled={isAnalyzing}
              >
                <Sparkles className="w-5 h-5" strokeWidth={2} />
              </button>
              <button className="p-2 text-gray-600 hover:bg-gray-200 hover:text-gray-900 rounded-full transition-all" disabled={isAnalyzing}>
                <Smile className="w-5 h-5" strokeWidth={2} />
              </button>
            </div>

            <button
              onClick={handlePost}
              disabled={content.trim().length === 0 || isAnalyzing || isDrafting}
              className={`px-5 py-2 rounded-full text-white text-[15px] font-bold transition-all ${
                content.trim().length === 0 || isAnalyzing || isDrafting
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
