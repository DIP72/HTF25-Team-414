import { Heart, MessageCircle, Repeat2, Send, MoreHorizontal } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";

interface PostProps {
  username: string;
  verified?: boolean;
  time: string;
  content: string;
  image?: string;
  replies: number;
  likes: number;
  avatar: string;
}

const Post = ({ username, verified, time, content, image, replies, likes, avatar }: PostProps) => {
  return (
    <article className="border-b border-border p-6 hover:bg-secondary/50 transition-colors">
      <div className="flex gap-3">
        <Avatar className="w-10 h-10">
          <AvatarImage src="" />
          <AvatarFallback className="bg-secondary text-foreground">{avatar}</AvatarFallback>
        </Avatar>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="font-medium text-foreground">{username}</span>
              {verified && (
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" className="text-accent">
                  <circle cx="8" cy="8" r="8" fill="currentColor"/>
                  <path d="M6 8L7.5 9.5L10 6.5" stroke="black" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
              <span className="text-muted-foreground text-sm">{time}</span>
            </div>
            <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
              <MoreHorizontal className="w-5 h-5" />
            </Button>
          </div>

          <p className="text-foreground text-[15px] leading-relaxed mb-3 whitespace-pre-wrap">{content}</p>

          {image && (
            <div className="mb-3 rounded-xl overflow-hidden border border-border">
              <img src={image} alt="Post content" className="w-full object-cover" />
            </div>
          )}

          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
              <Heart className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
              <MessageCircle className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
              <Repeat2 className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon" className="h-9 w-9 text-muted-foreground hover:text-foreground">
              <Send className="w-5 h-5" />
            </Button>
          </div>

          <div className="flex items-center gap-2 mt-2 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <Avatar className="w-4 h-4">
                <AvatarImage src="" />
                <AvatarFallback className="bg-muted text-[8px]">A</AvatarFallback>
              </Avatar>
              {replies} replies
            </span>
            <span>•</span>
            <span>{likes} likes</span>
          </div>
        </div>
      </div>
    </article>
  );
};

export default Post;
