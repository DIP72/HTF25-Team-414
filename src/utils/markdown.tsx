// src/utils/markdown.tsx (CREATE THIS FILE)
import React from "react";

export const parseMarkdown = (text: string): React.ReactNode => {
  if (!text) return null;

  // Split by line breaks but preserve them
  const lines = text.split('\n');
  
  return lines.map((line, lineIdx) => {
    if (!line.trim()) {
      return <br key={lineIdx} />;
    }

    // Process inline formatting
    let parts: (string | React.ReactNode)[] = [line];
    
    // Bold: **text**
    parts = parts.flatMap((part, idx) => {
      if (typeof part !== 'string') return part;
      const boldRegex = /\*\*(.+?)\*\*/g;
      const matches = [...part.matchAll(boldRegex)];
      if (matches.length === 0) return part;
      
      const result: (string | React.ReactNode)[] = [];
      let lastIndex = 0;
      matches.forEach((match, i) => {
        result.push(part.slice(lastIndex, match.index));
        result.push(<strong key={`${idx}-bold-${i}`}>{match[1]}</strong>);
        lastIndex = (match.index || 0) + match[0].length;
      });
      result.push(part.slice(lastIndex));
      return result;
    });

    // Italic: *text*
    parts = parts.flatMap((part, idx) => {
      if (typeof part !== 'string') return part;
      const italicRegex = /(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g;
      const matches = [...part.matchAll(italicRegex)];
      if (matches.length === 0) return part;
      
      const result: (string | React.ReactNode)[] = [];
      let lastIndex = 0;
      matches.forEach((match, i) => {
        result.push(part.slice(lastIndex, match.index));
        result.push(<em key={`${idx}-italic-${i}`}>{match[1]}</em>);
        lastIndex = (match.index || 0) + match[0].length;
      });
      result.push(part.slice(lastIndex));
      return result;
    });

    // Underline: __text__
    parts = parts.flatMap((part, idx) => {
      if (typeof part !== 'string') return part;
      const underlineRegex = /__(.+?)__/g;
      const matches = [...part.matchAll(underlineRegex)];
      if (matches.length === 0) return part;
      
      const result: (string | React.ReactNode)[] = [];
      let lastIndex = 0;
      matches.forEach((match, i) => {
        result.push(part.slice(lastIndex, match.index));
        result.push(<u key={`${idx}-underline-${i}`}>{match[1]}</u>);
        lastIndex = (match.index || 0) + match[0].length;
      });
      result.push(part.slice(lastIndex));
      return result;
    });

    return (
      <React.Fragment key={lineIdx}>
        {parts}
        {lineIdx < lines.length - 1 && <br />}
      </React.Fragment>
    );
  });
};
