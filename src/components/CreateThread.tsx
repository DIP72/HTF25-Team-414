import { useState, useRef } from 'react';
import { Image as ImageIcon, User, X, Sparkles, Loader2, ShieldAlert, Bold, Italic, Underline, Eye, AlertTriangle } from 'lucide-react';
import aiService from '@/services/aiService';
import { parseMarkdown } from '@/utils/markdown';
import { toast } from 'sonner';

interface CreateThreadProps {
  onPost: (
    content: string,
    images?: string[],
    labels?: string[],
    sentiment?: { label: string; confidence: number }
  ) => Promise<void> | void;  // ← Allow async
  currentUser: {
    username: string;
    handle: string;
    verified: boolean;
  };
}

const MAX_POST_CHARS = 1000;
const MIN_POST_CHARS = 20;

const CreateThread = ({ onPost, currentUser }: CreateThreadProps) => {
  const [content, setContent] = useState('');
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPosting, setIsPosting] = useState(false);  // ← New state
  const [analysis, setAnalysis] = useState<{
    sentiment?: { label: string; confidence: number };
    moderation?: { verdict: string; labels?: string[]; reason?: string };
    reviewFlag?: boolean;
  } | null>(null);
  const [showAIMenu, setShowAIMenu] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const compact = (t: string) => t.replace(/\s+/g, ' ').trim();
  const remaining = MAX_POST_CHARS - content.length;
  const overLimit = remaining < 0;
  const tooShort = compact(content).length < MIN_POST_CHARS && compact(content).length > 0;

  const analyzeTimeout = useRef<NodeJS.Timeout | null>(null);
  
  const handleContentChange = (newContent: string) => {
    setContent(newContent);
    setAnalysis(null);

    if (analyzeTimeout.current) {
      clearTimeout(analyzeTimeout.current);
    }

    const trimmed = compact(newContent);
    
    if (trimmed.length >= MIN_POST_CHARS) {
      analyzeTimeout.current = setTimeout(async () => {
        await analyzeContent(newContent);
      }, 1000);
    }
  };

  const analyzeContent = async (text: string) => {
    const base = compact(text);
    if (!base || base.length < MIN_POST_CHARS) return;

    setIsAnalyzing(true);
    try {
      const result = await aiService.analyzePost(base);
      const newAnalysis = {
        sentiment: result.sentiment,
        moderation: result.moderation,
        reviewFlag: result.review_flag,
      };
      setAnalysis(newAnalysis);

      if (result.moderation?.verdict === 'blocked') {
        toast.error(`Blocked: ${result.moderation.reason || 'Content violates guidelines'}`);
      }
      
      return newAnalysis;
    } catch (e: any) {
      console.error('Analysis failed:', e);
      
      if (e.message?.includes('403') || e.message?.includes('blocked')) {
        const blockedAnalysis = {
          sentiment: { label: 'NEUTRAL', confidence: 0.5 },
          moderation: { 
            verdict: 'blocked', 
            labels: ['violation'],
            reason: 'Content violates guidelines' 
          },
          reviewFlag: true
        };
        setAnalysis(blockedAnalysis);
        toast.error('Content blocked by moderation');
        return blockedAnalysis;
      }
      
      return {
        sentiment: { label: 'NEUTRAL', confidence: 0.5 },
        moderation: { verdict: 'safe' },
        reviewFlag: false,
      };
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handlePost = async () => {
    const base = compact(content);
    
    if (!base) {
      toast.error('Post cannot be empty');
      return;
    }

    if (base.length < MIN_POST_CHARS) {
      toast.error(`Post must be at least ${MIN_POST_CHARS} characters (currently ${base.length})`);
      return;
    }

    if (base.length > MAX_POST_CHARS) {
      toast.error('Post is too long');
      return;
    }

    // Ensure we have fresh analysis
    let finalAnalysis = analysis;
    if (!finalAnalysis?.moderation) {
      setIsAnalyzing(true);
      try {
        const res = await analyzeContent(base);
        if (res) finalAnalysis = res;
      } catch (e) {
        console.error('Pre-post analysis failed:', e);
      } finally {
        setIsAnalyzing(false);
      }
    }

    // Block if moderation says blocked
    if (finalAnalysis?.moderation?.verdict === 'blocked') {
      toast.error('Cannot post: Content violates guidelines');
      return;
    }

    // Post to database
    setIsPosting(true);
    try {
      await onPost(
        base,
        selectedImages.length > 0 ? selectedImages : undefined,
        finalAnalysis?.moderation?.labels || [],
        finalAnalysis?.sentiment || { label: 'NEUTRAL', confidence: 0.5 }
      );

      // Only reset on successful post
      setContent('');
      setSelectedImages([]);
      setAnalysis(null);
      
      // Don't show toast here - Index.tsx will handle it
    } catch (error) {
      console.error('Post failed:', error);
      // Error toast is handled by postsService
    } finally {
      setIsPosting(false);
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
      reader.onerror = () => loadedCount++;
      reader.readAsDataURL(file);
    });

    if (fileInputRef.current) fileInputRef.current.value = '';
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
      const cleaned = draft.replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '').replace(/\s+/g, ' ').trim();
      
      if (cleaned.length < MIN_POST_CHARS) {
        toast.warning(`Rewritten text is too short (${cleaned.length} chars). Keeping original.`);
        return;
      }
      
      setContent(cleaned);
      await analyzeContent(cleaned);
      toast.success('Content rewritten');
    } catch (e) {
      console.error('AI Rewrite failed:', e);
      toast.error('Rewrite failed');
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
      const cleaned = draft.replace(/[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '').replace(/\s+/g, ' ').trim();
      
      if (cleaned.length < MIN_POST_CHARS) {
        toast.warning(`Shortened text is too short (${cleaned.length} chars). Keeping original.`);
        return;
      }
      
      setContent(cleaned);
      await analyzeContent(cleaned);
      toast.success('Content shortened');
    } catch (e) {
      console.error('AI Condense failed:', e);
      toast.error('Condense failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const applyFormat = (formatType: 'bold' | 'italic' | 'underline') => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = content.substring(start, end);
    if (!selectedText) return;

    let formatted: string;
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

  const getSentimentBadge = () => {
    if (!analysis?.sentiment) return null;

    const { label, confidence } = analysis.sentiment;

    const normalize = (lab?: string) => {
      if (!lab) return 'neutral';
      const l = lab.toString().toLowerCase().trim();
      if (l.includes('pos') || l === '1' || l.includes('label_2')) return 'positive';
      if (l.includes('neg') || l === '0' || l.includes('label_0')) return 'negative';
      return 'neutral';
    };

    const norm = normalize(label);
    const sentimentStyles: Record<string, string> = {
      positive: 'bg-emerald-50 text-emerald-700 border-emerald-200',
      negative: 'bg-rose-50 text-rose-700 border-rose-200',
      neutral: 'bg-slate-50 text-slate-600 border-slate-200',
    };

    const style = sentimentStyles[norm] || 'bg-gray-50 text-gray-600 border-gray-200';
    const displayLabel = norm.charAt(0).toUpperCase() + norm.slice(1);
    const confidencePercent = Math.round((confidence ?? 0) * 100);

    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${style}`}>
        {displayLabel} {confidencePercent}%
      </span>
    );
  };

  const getModerationBadge = () => {
    if (!analysis?.moderation) return null;

    const { verdict, labels, reason } = analysis.moderation;

    if (verdict === 'blocked') {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-red-50 text-red-700 border border-red-200" title={reason || 'Blocked content'}>
          <ShieldAlert className="w-3 h-3" />
          blocked
        </span>
      );
    }

    if (verdict === 'flagged') {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200" title={reason || 'Needs review'}>
          <AlertTriangle className="w-3 h-3" />
          {labels?.[0] || 'flagged'}
        </span>
      );
    }

    return null;
  };

  const isBlocked = analysis?.moderation?.verdict === 'blocked';
  const canPost = !isBlocked && !tooShort && !overLimit && compact(content).length >= MIN_POST_CHARS && !isPosting;

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-4 mb-4 shadow-sm">
      <div className="flex gap-3">
        <div className="w-10 h-10 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 text-gray-600" strokeWidth={2} />
        </div>

        <div className="flex-1">
          <textarea
            ref={textareaRef}
            placeholder={`What's happening? (min ${MIN_POST_CHARS} characters)`}
            className={`w-full bg-transparent text-gray-900 text-base placeholder:text-gray-500 border-none outline-none resize-none min-h-[100px] mb-2 ${
              overLimit || isBlocked || tooShort ? 'ring-2 ring-red-300 rounded-lg p-2' : ''
            }`}
            value={content}
            onChange={(e) => handleContentChange(e.target.value)}
            disabled={isAnalyzing || isProcessing || isPosting}
          />

          {selectedImages.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {selectedImages.map((img, idx) => (
                <div key={idx} className="relative group">
                  <img src={img} alt={`Upload ${idx + 1}`} className="h-20 w-20 object-cover rounded-lg" />
                  <button
                    onClick={() => removeImage(idx)}
                    className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {(analysis || isAnalyzing) && (
            <div className="flex items-center gap-2 mb-3 flex-wrap">
              {isAnalyzing && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-blue-50 text-blue-600 border border-blue-200">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  analyzing...
                </span>
              )}
              {!isAnalyzing && getSentimentBadge()}
              {!isAnalyzing && getModerationBadge()}
            </div>
          )}

          {showPreview && content.trim() && (
            <div className="mb-3 p-3 bg-gray-50 rounded-xl border border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-gray-500">Preview</span>
                <button
                  onClick={() => setShowPreview(false)}
                  className="text-xs text-blue-600 hover:underline"
                >
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
                onClick={() => applyFormat('bold')}
                className="p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Bold"
                disabled={isAnalyzing || isProcessing || isPosting}
              >
                <Bold className="w-4 h-4" strokeWidth={2.5} />
              </button>
              <button
                onClick={() => applyFormat('italic')}
                className="p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Italic"
                disabled={isAnalyzing || isProcessing || isPosting}
              >
                <Italic className="w-4 h-4" strokeWidth={2.5} />
              </button>
              <button
                onClick={() => applyFormat('underline')}
                className="p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Underline"
                disabled={isAnalyzing || isProcessing || isPosting}
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
                disabled={isAnalyzing || isProcessing || isPosting}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-all disabled:opacity-40"
                title="Upload images"
                disabled={isAnalyzing || isProcessing || isPosting}
              >
                <ImageIcon className="w-5 h-5" strokeWidth={2} />
              </button>

              <button
                onClick={() => setShowPreview(!showPreview)}
                className={`p-2 rounded-lg transition-all ${showPreview ? 'bg-blue-100 text-blue-600' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'}`}
                title="Toggle preview"
                disabled={!content.trim()}
              >
                <Eye className="w-5 h-5" strokeWidth={2} />
              </button>

              <div className="w-px h-6 bg-gray-300 mx-1" />

              <div className="relative" onClick={(e) => e.stopPropagation()}>
                <button
                  onClick={() => setShowAIMenu(!showAIMenu)}
                  disabled={!content.trim() || isAnalyzing || isProcessing || isPosting}
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
            </div>

            <div className="flex items-center gap-3">
              <div className="text-xs text-gray-500">
                {overLimit ? (
                  <span className="text-red-600 font-medium">Over by {Math.abs(remaining)}</span>
                ) : tooShort ? (
                  <span className="text-amber-600 font-medium">
                    Need {MIN_POST_CHARS - compact(content).length} more
                  </span>
                ) : (
                  <span className={remaining < 100 ? 'text-amber-600' : ''}>{remaining}</span>
                )}
              </div>

              <button
                onClick={handlePost}
                disabled={!canPost || isAnalyzing || isProcessing || isPosting}
                className={`px-5 py-2 rounded-full text-white text-[15px] font-bold transition-all ${
                  !canPost || isAnalyzing || isProcessing || isPosting
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-black hover:bg-gray-800'
                }`}
              >
                {isPosting ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Posting...
                  </span>
                ) : isAnalyzing ? (
                  'Analyzing...'
                ) : isBlocked ? (
                  'Blocked'
                ) : tooShort ? (
                  'Too Short'
                ) : (
                  'Post'
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreateThread;
