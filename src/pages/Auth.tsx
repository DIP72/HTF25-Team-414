import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { supabase } from '@/lib/supabase'
import { Sparkles } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

const Auth = () => {
  const { user, profile, signInWithGoogle } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [handle, setHandle] = useState('')
  const [isCreating, setIsCreating] = useState(false)

  useEffect(() => {
    if (user && profile) {
      navigate('/')
    }
  }, [user, profile, navigate])

  const handleGoogleAuth = async () => {
    await signInWithGoogle()
  }

  const handleCreateProfile = async () => {
    if (!user || !username.trim() || !handle.trim()) return

    setIsCreating(true)
    try {
      const cleanHandle = handle.trim().toLowerCase().replace(/[^a-z0-9_]/g, '')
      
      const { error } = await supabase
        .from('profiles')
        .insert({
          id: user.id,
          email: user.email!,
          username: username.trim(),
          handle: cleanHandle,
          role: 'user',
          verified: false
        })

      if (error) {
        if (error.code === '23505') {
          alert('This username or handle is already taken. Please choose another.')
        } else {
          throw error
        }
      } else {
        window.location.href = '/'
      }
    } catch (error) {
      console.error('Error creating profile:', error)
      alert('Failed to create profile. Please try again.')
    } finally {
      setIsCreating(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl shadow-2xl p-8 max-w-md w-full">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-3">
            <div className="p-3 bg-blue-100 rounded-2xl">
              <Sparkles className="w-8 h-8 text-blue-600" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">ThreadsAI</h1>
          <p className="text-gray-600">AI-powered social conversations</p>
        </div>

        {!user ? (
          <div className="space-y-4">
            <button
              onClick={handleGoogleAuth}
              className="w-full flex items-center justify-center gap-3 bg-white border-2 border-gray-300 hover:border-blue-400 hover:shadow-md rounded-xl px-6 py-4 font-semibold text-gray-700 transition-all"
            >
              <svg className="w-6 h-6" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              Continue with Google
            </button>

            <div className="text-center text-xs text-gray-500 mt-4">
              By continuing, you agree to our Terms of Service and Privacy Policy
            </div>
          </div>
        ) : !profile && (
          <div className="space-y-5">
            <div className="text-center mb-6">
              <h2 className="text-xl font-bold text-gray-900 mb-1">Complete Your Profile</h2>
              <p className="text-sm text-gray-600">Choose your username and handle</p>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Display Name
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="John Doe"
                maxLength={50}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Username (Handle)
              </label>
              <div className="flex items-center gap-2 border-2 border-gray-200 rounded-xl px-4 py-3 focus-within:border-blue-500 transition-colors">
                <span className="text-gray-500 font-medium">@</span>
                <input
                  type="text"
                  value={handle}
                  onChange={(e) => setHandle(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))}
                  placeholder="johndoe"
                  maxLength={20}
                  className="flex-1 outline-none"
                />
              </div>
              <p className="text-xs text-gray-500 mt-2">Only lowercase letters, numbers, and underscores</p>
            </div>

            <button
              onClick={handleCreateProfile}
              disabled={!username.trim() || !handle.trim() || isCreating}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white font-bold py-4 rounded-xl transition-all disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
            >
              {isCreating ? 'Creating...' : 'Create Profile'}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default Auth
