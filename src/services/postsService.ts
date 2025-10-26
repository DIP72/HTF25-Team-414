import { supabase, Post, Reply, PostWithInteractions } from '@/lib/supabase'
import { toast } from 'sonner'

export const postsService = {
  // Fetch all posts with user interactions
  async fetchPosts(userId?: string): Promise<PostWithInteractions[]> {
    try {
      const { data: posts, error } = await supabase
        .from('posts')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error
      if (!posts) return []

      // Fetch user's interactions if logged in
      if (userId) {
        const [likesRes, repostsRes, bookmarksRes] = await Promise.all([
          supabase.from('likes').select('post_id').eq('user_id', userId),
          supabase.from('reposts').select('post_id').eq('user_id', userId),
          supabase.from('bookmarks').select('post_id').eq('user_id', userId),
        ])

        const likedPostIds = new Set(likesRes.data?.map(l => l.post_id) || [])
        const repostedPostIds = new Set(repostsRes.data?.map(r => r.post_id) || [])
        const bookmarkedPostIds = new Set(bookmarksRes.data?.map(b => b.post_id) || [])

        return posts.map(post => ({
          ...post,
          isLiked: likedPostIds.has(post.id),
          isReposted: repostedPostIds.has(post.id),
          isBookmarked: bookmarkedPostIds.has(post.id),
        }))
      }

      return posts
    } catch (error: any) {
      console.error('Error fetching posts:', error)
      toast.error('Failed to load posts')
      return []
    }
  },

  // Create a new post with auto-profile creation
  async createPost(postData: {
    content: string
    images?: string[]
    sentiment_label?: string
    sentiment_confidence?: number
    flag_label?: string
  }): Promise<Post | null> {
    try {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        toast.error('You must be logged in to post')
        return null
      }

      console.log('Creating post for user:', user.id)

      // Try to fetch profile
      const { data: profile, error: profileError } = await supabase
        .from('profiles')
        .select('username, handle, role')
        .eq('id', user.id)
        .single()

      let userProfile = profile

      // If profile doesn't exist, create one
      if (profileError || !profile) {
        console.log('Profile not found, creating new profile')
        
        const username = user.user_metadata?.username || user.email?.split('@')[0] || 'User'
        const handle = user.user_metadata?.handle || user.email?.split('@')[0] || 'user'
        
        const { data: newProfile, error: createError } = await supabase
          .from('profiles')
          .insert({
            id: user.id,
            email: user.email || '',
            username: username,
            handle: handle,
            role: 'user'
          })
          .select('username, handle, role')
          .single()

        if (createError) {
          console.error('Failed to create profile:', createError)
          toast.error('Please set up your profile first')
          return null
        }

        userProfile = newProfile
      }

      if (!userProfile) {
        toast.error('Could not load user profile')
        return null
      }

      console.log('Using profile:', userProfile)

      const insertData = {
        user_id: user.id,
        username: userProfile.username,
        handle: userProfile.handle,
        role: userProfile.role,
        content: postData.content,
        images: postData.images || null,
        sentiment_label: postData.sentiment_label || null,
        sentiment_confidence: postData.sentiment_confidence || null,
        flag_label: postData.flag_label || null,
      }

      console.log('Inserting post:', insertData)

      const { data, error } = await supabase
        .from('posts')
        .insert(insertData)
        .select()
        .single()

      if (error) {
        console.error('Insert error:', error)
        toast.error(`Failed: ${error.message}`)
        return null
      }

      console.log('Post created:', data)
      toast.success('Post created!')
      return data
    } catch (error: any) {
      console.error('Error creating post:', error)
      toast.error('Failed to create post')
      return null
    }
  },

  // Update post
  async updatePost(postId: string, content: string): Promise<boolean> {
    try {
      const { error } = await supabase
        .from('posts')
        .update({ 
          content, 
          updated_at: new Date().toISOString() 
        })
        .eq('id', postId)

      if (error) throw error
      
      toast.success('Post updated!')
      return true
    } catch (error: any) {
      console.error('Error updating post:', error)
      toast.error('Failed to update post')
      return false
    }
  },

  // Delete post
  async deletePost(postId: string): Promise<boolean> {
    try {
      const { error } = await supabase
        .from('posts')
        .delete()
        .eq('id', postId)

      if (error) throw error
      
      toast.success('Post deleted!')
      return true
    } catch (error: any) {
      console.error('Error deleting post:', error)
      toast.error('Failed to delete post')
      return false
    }
  },

  // Toggle like with immediate DB sync
  async toggleLike(postId: string): Promise<boolean> {
    try {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        toast.error('You must be logged in to like')
        return false
      }

      const { data: existing } = await supabase
        .from('likes')
        .select()
        .eq('user_id', user.id)
        .eq('post_id', postId)
        .maybeSingle()

      if (existing) {
        // Unlike
        const { error } = await supabase
          .from('likes')
          .delete()
          .eq('id', existing.id)
        
        if (error) throw error
        console.log('Unliked post:', postId)
      } else {
        // Like
        const { error } = await supabase
          .from('likes')
          .insert({ user_id: user.id, post_id: postId })
        
        if (error) throw error
        console.log('Liked post:', postId)
      }

      return true
    } catch (error: any) {
      console.error('Error toggling like:', error)
      toast.error('Failed to update like')
      return false
    }
  },

  // Get fresh like count from database
  async getLikeCount(postId: string): Promise<number> {
    try {
      const { data, error } = await supabase
        .from('posts')
        .select('likes_count')
        .eq('id', postId)
        .single()

      if (error) throw error
      return data?.likes_count || 0
    } catch (error) {
      console.error('Error fetching like count:', error)
      return 0
    }
  },

  // Toggle repost
  async toggleRepost(postId: string): Promise<boolean> {
    try {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        toast.error('You must be logged in to repost')
        return false
      }

      const { data: existing } = await supabase
        .from('reposts')
        .select()
        .eq('user_id', user.id)
        .eq('post_id', postId)
        .maybeSingle()

      if (existing) {
        // Unrepost
        const { error } = await supabase
          .from('reposts')
          .delete()
          .eq('id', existing.id)
        
        if (error) throw error
        console.log('Unreposted post:', postId)
      } else {
        // Repost
        const { error } = await supabase
          .from('reposts')
          .insert({ user_id: user.id, post_id: postId })
        
        if (error) throw error
        console.log('Reposted post:', postId)
      }

      return true
    } catch (error: any) {
      console.error('Error toggling repost:', error)
      toast.error('Failed to update repost')
      return false
    }
  },

  // Get fresh repost count from database
  async getRepostCount(postId: string): Promise<number> {
    try {
      const { data, error } = await supabase
        .from('posts')
        .select('reposts_count')
        .eq('id', postId)
        .single()

      if (error) throw error
      return data?.reposts_count || 0
    } catch (error) {
      console.error('Error fetching repost count:', error)
      return 0
    }
  },

  // Toggle bookmark
  async toggleBookmark(postId: string): Promise<boolean> {
    try {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        toast.error('You must be logged in to bookmark')
        return false
      }

      const { data: existing } = await supabase
        .from('bookmarks')
        .select()
        .eq('user_id', user.id)
        .eq('post_id', postId)
        .maybeSingle()

      if (existing) {
        await supabase.from('bookmarks').delete().eq('id', existing.id)
      } else {
        await supabase.from('bookmarks').insert({ user_id: user.id, post_id: postId })
      }

      return true
    } catch (error: any) {
      console.error('Error toggling bookmark:', error)
      toast.error('Failed to update bookmark')
      return false
    }
  },

  // Fetch replies for a post
  async fetchReplies(postId: string): Promise<Reply[]> {
    try {
      const { data, error } = await supabase
        .from('replies')
        .select('*')
        .eq('post_id', postId)
        .order('created_at', { ascending: true })

      if (error) throw error
      return data || []
    } catch (error: any) {
      console.error('Error fetching replies:', error)
      return []
    }
  },

  // Create reply
  async createReply(
    postId: string,
    content: string,
    sentiment_label?: string,
    sentiment_confidence?: number
  ): Promise<Reply | null> {
    try {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        toast.error('You must be logged in to reply')
        return null
      }

      const { data: profile } = await supabase
        .from('profiles')
        .select('username, handle, role')
        .eq('id', user.id)
        .single()

      if (!profile) {
        toast.error('Profile not found')
        return null
      }

      const { data, error } = await supabase
        .from('replies')
        .insert({
          post_id: postId,
          user_id: user.id,
          username: profile.username,
          handle: profile.handle,
          role: profile.role,
          content,
          sentiment_label: sentiment_label || null,
          sentiment_confidence: sentiment_confidence || null,
        })
        .select()
        .single()

      if (error) throw error
      
      toast.success('Reply posted!')
      return data
    } catch (error: any) {
      console.error('Error creating reply:', error)
      toast.error('Failed to post reply')
      return null
    }
  },

  // Increment views (one per user/session)
  async incrementViews(postId: string): Promise<void> {
    try {
      // Session-based view tracking
      const viewKey = `post_view_${postId}`
      const hasViewed = sessionStorage.getItem(viewKey)
      
      if (!hasViewed) {
        // Use RPC function for atomic increment
        const { error } = await supabase.rpc('increment_post_views_simple', {
          post_uuid: postId
        })
        
        if (error) {
          console.warn('Failed to increment views:', error)
        } else {
          sessionStorage.setItem(viewKey, 'true')
        }
      }
    } catch (error) {
      console.warn('Failed to increment views:', error)
    }
  },

  // Subscribe to real-time updates
  subscribeToPostUpdates(callback: (payload: any) => void) {
    const channel = supabase
      .channel('posts-changes')
      .on(
        'postgres_changes',
        { event: '*', schema: 'public', table: 'posts' },
        callback
      )
      .subscribe()

    return channel
  },
}
