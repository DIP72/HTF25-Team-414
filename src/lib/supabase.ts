import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

export interface UserProfile {
  id: string
  email: string
  username: string
  handle: string
  role: 'user' | 'admin' | 'verified'
  verified: boolean
  created_at?: string
}
