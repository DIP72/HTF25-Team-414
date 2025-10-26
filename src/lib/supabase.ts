import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Type definitions
export interface UserProfile {
  id: string
  email: string
  username: string
  handle: string
  role: 'user' | 'admin' | 'verified'
  created_at?: string
}

export interface Post {
  id: string
  user_id: string
  username: string
  handle: string
  role: 'user' | 'admin' | 'verified'  // ← Changed from verified: boolean
  content: string
  images: string[] | null
  sentiment_label: string | null
  sentiment_confidence: number | null
  flag_label: string | null
  likes_count: number
  reposts_count: number
  views_count: number
  replies_count: number
  created_at: string
  updated_at: string
}

export interface Reply {
  id: string
  post_id: string
  user_id: string
  username: string
  handle: string
  role: 'user' | 'admin' | 'verified'  // ← Changed from verified: boolean
  content: string
  images: string[] | null
  sentiment_label: string | null
  sentiment_confidence: number | null
  likes_count: number
  views_count: number
  created_at: string
}

export interface PostWithInteractions extends Post {
  isLiked?: boolean
  isReposted?: boolean
  isBookmarked?: boolean
  replies?: Reply[]
}

export type Database = {
  public: {
    Tables: {
      posts: {
        Row: Post
        Insert: Omit<Post, 'id' | 'likes_count' | 'reposts_count' | 'views_count' | 'replies_count' | 'created_at' | 'updated_at'>
        Update: Partial<Pick<Post, 'content' | 'images' | 'updated_at'>>
      }
      replies: {
        Row: Reply
        Insert: Omit<Reply, 'id' | 'likes_count' | 'views_count' | 'created_at'>
        Update: Partial<Pick<Reply, 'content'>>
      }
    }
  }
}
