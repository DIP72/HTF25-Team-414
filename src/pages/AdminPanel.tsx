import { useState, useEffect } from 'react'
import { supabase, UserProfile } from '@/lib/supabase'
import { useAuth } from '@/contexts/AuthContext'
import { Shield, User as UserIcon, Search, ArrowLeft } from 'lucide-react'
import Layout from '@/components/Layout'
import { useNavigate } from 'react-router-dom'

const AdminPanel = () => {
  const { profile } = useAuth()
  const navigate = useNavigate()
  const [users, setUsers] = useState<UserProfile[]>([])
  const [filteredUsers, setFilteredUsers] = useState<UserProfile[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    if (profile?.role !== 'admin') {
      navigate('/')
      return
    }
    fetchUsers()
  }, [profile, navigate])

  useEffect(() => {
    if (searchQuery.trim() === '') {
      setFilteredUsers(users)
    } else {
      const query = searchQuery.toLowerCase()
      setFilteredUsers(
        users.filter(
          (user) =>
            user.username.toLowerCase().includes(query) ||
            user.handle.toLowerCase().includes(query) ||
            user.email.toLowerCase().includes(query)
        )
      )
    }
  }, [searchQuery, users])

  const fetchUsers = async () => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error
      setUsers(data || [])
      setFilteredUsers(data || [])
    } catch (error) {
      console.error('Error fetching users:', error)
    } finally {
      setLoading(false)
    }
  }

  const updateUserRole = async (userId: string, newRole: 'user' | 'admin' | 'verified') => {
    try {
      const { error } = await supabase
        .from('profiles')
        .update({ role: newRole })
        .eq('id', userId)

      if (error) throw error
      
      setUsers(prev => prev.map(u => u.id === userId ? { ...u, role: newRole } : u))
    } catch (error) {
      console.error('Error updating role:', error)
      alert('Failed to update role')
    }
  }

  if (profile?.role !== 'admin') return null

  return (
    <Layout>
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10 p-4">
        <div className="flex items-center gap-3 mb-4">
          <button
            onClick={() => navigate('/')}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <Shield className="w-6 h-6 text-red-500" />
            <h1 className="text-xl font-bold text-gray-900">User Management</h1>
          </div>
        </div>

        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search users by name, handle, or email..."
            className="w-full bg-gray-100 rounded-xl py-3 pl-12 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
          />
        </div>

        <div className="flex items-center gap-4 mt-3 text-sm text-gray-600">
          <span>Total Users: <strong>{users.length}</strong></span>
          <span>Showing: <strong>{filteredUsers.length}</strong></span>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent mx-auto mb-4"></div>
          <p className="text-gray-600">Loading users...</p>
        </div>
      ) : (
        <div className="p-4 space-y-3">
          {filteredUsers.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              No users found matching "{searchQuery}"
            </div>
          ) : (
            filteredUsers.map((user) => (
              <div
                key={user.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-2xl hover:bg-gray-100 transition-colors border border-gray-200"
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                    <UserIcon className="w-6 h-6 text-gray-600" strokeWidth={2} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 mb-1">
                      <p className="text-sm font-bold text-gray-900 truncate">{user.username}</p>
                      {user.role === 'verified' && (
                        <svg width="16" height="16" viewBox="0 0 22 22" className="text-blue-500 flex-shrink-0">
                          <path fill="currentColor" d="M20.396 11c-.018-.646-.215-1.275-.57-1.816-.354-.54-.852-.972-1.438-1.246.223-.607.27-1.264.14-1.897-.131-.634-.437-1.218-.882-1.687-.47-.445-1.053-.75-1.687-.882-.633-.13-1.29-.083-1.897.14-.273-.587-.704-1.086-1.245-1.44S11.647 1.62 11 1.604c-.646.017-1.273.213-1.813.568s-.969.854-1.24 1.44c-.608-.223-1.267-.272-1.902-.14-.635.13-1.22.436-1.69.882-.445.47-.749 1.055-.878 1.688-.13.633-.08 1.29.144 1.896-.587.274-1.087.705-1.443 1.245-.356.54-.555 1.17-.574 1.817.02.647.218 1.276.574 1.817.356.54.856.972 1.443 1.245-.224.606-.274 1.263-.144 1.896.13.634.433 1.218.877 1.688.47.443 1.054.747 1.687.878.633.132 1.29.084 1.897-.136.274.586.705 1.084 1.246 1.439.54.354 1.17.551 1.816.569.647-.016 1.276-.213 1.817-.567s.972-.854 1.245-1.44c.604.239 1.266.296 1.903.164.636-.132 1.22-.447 1.68-.907.46-.46.776-1.044.908-1.681s.075-1.299-.165-1.903c.586-.274 1.084-.705 1.439-1.246.354-.54.551-1.17.569-1.816zM9.662 14.85l-3.429-3.428 1.293-1.302 2.072 2.072 4.4-4.794 1.347 1.246z"/>
                        </svg>
                      )}
                      {user.role === 'admin' && (
                        <svg width="16" height="16" viewBox="0 0 22 22" className="text-red-500 flex-shrink-0">
                          <path fill="currentColor" d="M20.396 11c-.018-.646-.215-1.275-.57-1.816-.354-.54-.852-.972-1.438-1.246.223-.607.27-1.264.14-1.897-.131-.634-.437-1.218-.882-1.687-.47-.445-1.053-.75-1.687-.882-.633-.13-1.29-.083-1.897.14-.273-.587-.704-1.086-1.245-1.44S11.647 1.62 11 1.604c-.646.017-1.273.213-1.813.568s-.969.854-1.24 1.44c-.608-.223-1.267-.272-1.902-.14-.635.13-1.22.436-1.69.882-.445.47-.749 1.055-.878 1.688-.13.633-.08 1.29.144 1.896-.587.274-1.087.705-1.443 1.245-.356.54-.555 1.17-.574 1.817.02.647.218 1.276.574 1.817.356.54.856.972 1.443 1.245-.224.606-.274 1.263-.144 1.896.13.634.433 1.218.877 1.688.47.443 1.054.747 1.687.878.633.132 1.29.084 1.897-.136.274.586.705 1.084 1.246 1.439.54.354 1.17.551 1.816.569.647-.016 1.276-.213 1.817-.567s.972-.854 1.245-1.44c.604.239 1.266.296 1.903.164.636-.132 1.22-.447 1.68-.907.46-.46.776-1.044.908-1.681s.075-1.299-.165-1.903c.586-.274 1.084-.705 1.439-1.246.354-.54.551-1.17.569-1.816zM15.5 9.5l-5 5-1.5 1.5-1.5-1.5-2-2 1.5-1.5 2 2 5-5 1.5 1.5z"/>
                        </svg>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 truncate mb-0.5">@{user.handle}</p>
                    <p className="text-xs text-gray-400 truncate">{user.email}</p>
                  </div>
                </div>
                
                <div className="flex gap-2 flex-shrink-0">
                  <button
                    onClick={() => updateUserRole(user.id, 'user')}
                    className={`px-3 py-2 text-xs font-semibold rounded-lg transition-all ${
                      user.role === 'user'
                        ? 'bg-gray-900 text-white shadow-md'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                    disabled={user.id === profile.id}
                  >
                    User
                  </button>
                  <button
                    onClick={() => updateUserRole(user.id, 'verified')}
                    className={`px-3 py-2 text-xs font-semibold rounded-lg transition-all ${
                      user.role === 'verified'
                        ? 'bg-blue-500 text-white shadow-md'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                    disabled={user.id === profile.id}
                  >
                    Verified
                  </button>
                  <button
                    onClick={() => updateUserRole(user.id, 'admin')}
                    className={`px-3 py-2 text-xs font-semibold rounded-lg transition-all ${
                      user.role === 'admin'
                        ? 'bg-red-500 text-white shadow-md'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                    disabled={user.id === profile.id}
                  >
                    Admin
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </Layout>
  )
}

export default AdminPanel
