import React from "react";

export const parseMarkdown = (text: string): React.ReactNode[] => {
  const parts: React.ReactNode[] = [];
  const lines = text.split("\n");
  let key = 0;

  // Inline formatting patterns
  const inlinePatterns = [
    { regex: /\*\*(.+?)\*\*/g, tag: "strong" }, // **bold**
    { regex: /__(.+?)__/g, tag: "u" },          // __underline__
    { regex: /\*(.+?)\*/g, tag: "em" },         // *italic*
    { regex: /`(.+?)`/g, tag: "code" },         // `inline code`
  ];

  const parseInline = (str: string) => {
    let remaining = str;
    const inlineParts: React.ReactNode[] = [];
    let k = 0;

    while (remaining.length > 0) {
      let earliest: { index: number; length: number; content: string; tag: string } | null = null;

      for (const p of inlinePatterns) {
        const match = p.regex.exec(remaining);
        if (match && (!earliest || match.index < earliest.index)) {
          earliest = { index: match.index, length: match[0].length, content: match[1], tag: p.tag };
        }
        p.regex.lastIndex = 0;
      }

      if (earliest) {
        if (earliest.index > 0) inlineParts.push(<React.Fragment key={k++}>{remaining.substring(0, earliest.index)}</React.Fragment>);
        const Tag = earliest.tag as keyof JSX.IntrinsicElements;
        inlineParts.push(<Tag key={k++}>{earliest.content}</Tag>);
        remaining = remaining.substring(earliest.index + earliest.length);
      } else {
        inlineParts.push(<React.Fragment key={k++}>{remaining}</React.Fragment>);
        break;
      }
    }

    return inlineParts;
  };

  let inList = false;
  let listItems: React.ReactNode[] = [];
  let orderedList = false;

  const flushList = () => {
    if (!inList) return;
    const Tag = orderedList ? "ol" : "ul";
    parts.push(<Tag key={key++} className="ml-4 mb-2 list-disc list-inside">{listItems}</Tag>);
    listItems = [];
    inList = false;
    orderedList = false;
  };

  lines.forEach((line) => {
    // Headers
    const headerMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headerMatch) {
      flushList();
      const level = headerMatch[1].length;
      const content = headerMatch[2];
      const Tag = `h${level}` as keyof JSX.IntrinsicElements;
      parts.push(<Tag key={key++}>{parseInline(content)}</Tag>);
      return;
    }

    // Lists
    const ulMatch = line.match(/^[-*]\s+(.+)$/);
    const olMatch = line.match(/^(\d+)\.\s+(.+)$/);

    if (ulMatch || olMatch) {
      const content = ulMatch ? ulMatch[1] : olMatch![2];
      if (!inList) {
        inList = true;
        orderedList = !!olMatch;
      }
      listItems.push(<li key={listItems.length}>{parseInline(content)}</li>);
      return;
    }

    // Normal paragraph
    flushList();
    if (line.trim() === "") {
      parts.push(<br key={key++} />);
    } else {
      parts.push(<p key={key++}>{parseInline(line)}</p>);
    }
  });

  flushList();
  return parts;
};
