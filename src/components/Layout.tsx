// src/components/Layout.tsx
import { Home, Search, Bell, Mail, User, Bookmark, MoreHorizontal, X } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useState } from "react";

const Layout = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { icon: Home, label: "Home", path: "/" },
    { icon: Search, label: "Explore", path: "/explore" },
    { icon: Bell, label: "Notifications", path: "/notifications" },
    { icon: Mail, label: "Messages", path: "/messages" },
    { icon: Bookmark, label: "Bookmarks", path: "/bookmarks" },
    { icon: User, label: "Profile", path: "/profile" },
  ];

  const whoToFollow = [
    { name: "Dan Miluanton", handle: "@dan_miluanton", verified: false },
    { name: "Amilia Gonzales", handle: "@amiliag_7s1g", verified: false },
    { name: "Kim Chiushiu", handle: "@kim_Him_chiuze", verified: false },
  ];

  const exploreTopics = [
    { title: "AI Moderation Discussion", subtitle: "Trending Now • Technology" },
    { title: "React 19 Features Released", subtitle: "Trending Now • Development" },
    { title: "Hackathon Projects Showcase", subtitle: "Trending Now • Events" },
  ];

  return (
    <div className="min-h-screen bg-white pb-16 md:pb-0">
      {/* Mobile Header */}
      <header className="md:hidden sticky top-0 z-40 bg-white border-b border-gray-200 px-4 py-3">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <h1 className="text-xl font-bold text-gray-900 tracking-tight">ThreadsAI</h1>
          </Link>
          <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="p-2 hover:bg-gray-100 rounded-full transition-colors">
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <MoreHorizontal className="w-6 h-6" />}
          </button>
        </div>
      </header>

      {/* Mobile Dropdown Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed inset-x-0 top-[57px] bottom-16 bg-white z-30 overflow-y-auto border-b border-gray-200">
          <div className="p-4 space-y-4">
            {/* User Profile */}
            <div className="bg-gray-50 rounded-2xl p-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center">
                  <User className="w-6 h-6 text-gray-600" strokeWidth={2} />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-bold text-gray-900">Your Name</p>
                  <p className="text-sm text-gray-500">@yourusername</p>
                </div>
              </div>
            </div>

            {/* Explore Topics */}
            <div className="bg-gray-50 rounded-2xl overflow-hidden">
              <div className="p-4">
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-bold text-gray-900">Explore</h2>
                  <span className="text-xs font-semibold text-blue-600 bg-blue-100 px-2 py-1 rounded-full">Beta</span>
                </div>
              </div>
              <div>
                {exploreTopics.map((topic, idx) => (
                  <div key={idx} className="px-4 py-3 border-t border-gray-200">
                    <p className="text-xs text-gray-500 mb-1">{topic.subtitle}</p>
                    <p className="text-sm font-bold text-gray-900">{topic.title}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Who to Follow */}
            <div className="bg-gray-50 rounded-2xl overflow-hidden">
              <div className="p-4">
                <h2 className="text-lg font-bold text-gray-900">Who to follow</h2>
              </div>
              <div>
                {whoToFollow.map((user, idx) => (
                  <div key={idx} className="px-4 py-3 border-t border-gray-200">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <div className="w-10 h-10 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                          <User className="w-5 h-5 text-gray-600" strokeWidth={2} />
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-bold text-gray-900 truncate">{user.name}</p>
                          <p className="text-sm text-gray-500 truncate">{user.handle}</p>
                        </div>
                      </div>
                      <button className="px-4 py-1.5 bg-black hover:bg-gray-800 text-white text-sm font-bold rounded-full transition-colors flex-shrink-0">
                        Follow
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-[1280px] mx-auto md:flex md:gap-6 md:p-6">
        {/* Left Sidebar - Desktop Only */}
        <aside className="hidden lg:block w-[275px] flex-shrink-0 sticky top-6 h-fit">
          <div className="bg-gray-50 rounded-2xl p-4">
            <Link to="/" className="block mb-4 px-2">
              <h1 className="text-2xl font-bold text-gray-900 tracking-tight">ThreadsAI</h1>
            </Link>

            <nav className="space-y-2">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link key={item.path} to={item.path}>
                    <div className={`relative w-full flex items-center gap-4 px-4 py-3 rounded-full transition-colors duration-200 text-lg ${isActive ? "font-bold" : "font-normal"}`}>
                      <div className={`absolute inset-0 rounded-full transition-colors duration-200 ${isActive ? "bg-gray-200" : ""}`} />
                      <div className="absolute inset-0 rounded-full bg-gray-200 opacity-0 hover:opacity-100 transition-opacity duration-200" />
                      <item.icon className="w-6 h-6 relative z-10" strokeWidth={isActive ? 2.5 : 2} />
                      <span className="relative z-10">{item.label}</span>
                    </div>
                  </Link>
                );
              })}
            </nav>

            <button className="w-full mt-4 bg-black hover:bg-gray-800 text-white py-3 rounded-full font-bold text-[15px] transition-colors duration-200">
              Post
            </button>

            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="relative rounded-full">
                <div className="absolute inset-0 rounded-full bg-gray-200 opacity-0 hover:opacity-100 transition-opacity duration-200" />
                <button className="relative z-10 w-full flex items-center gap-3 p-3">
                  <div className="w-10 h-10 rounded-full bg-gray-300 flex items-center justify-center">
                    <User className="w-5 h-5 text-gray-600" strokeWidth={2} />
                  </div>
                  <div className="flex-1 text-left">
                    <p className="text-sm font-bold text-gray-900">Your Name</p>
                    <p className="text-sm text-gray-500">@yourusername</p>
                  </div>
                  <MoreHorizontal className="w-5 h-5 text-gray-900" />
                </button>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 min-w-0 md:max-w-[600px] px-0 md:px-0">
          {children}
        </main>

        {/* Right Sidebar - Desktop Only */}
        <aside className="hidden xl:block w-[350px] flex-shrink-0 space-y-4 sticky top-6 h-fit">
          {/* Search */}
          <div className="bg-gray-50 rounded-2xl p-4">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
              <input
                type="text"
                placeholder="Search"
                className="w-full bg-gray-200 rounded-full py-3 pl-14 pr-4 text-[15px] focus:outline-none focus:bg-white focus:ring-2 focus:ring-black transition-all duration-200"
              />
            </div>
          </div>

          {/* Explore */}
          <div className="bg-gray-50 rounded-2xl overflow-hidden">
            <div className="p-4">
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-bold text-gray-900">Explore</h2>
                <span className="text-xs font-semibold text-blue-600 bg-blue-100 px-2 py-1 rounded-full">Beta</span>
              </div>
            </div>
            <div>
              {exploreTopics.map((topic, idx) => (
                <div key={idx} className="relative px-4 py-3 cursor-pointer border-t border-gray-200">
                  <div className="absolute inset-0 bg-gray-200 opacity-0 hover:opacity-100 transition-opacity duration-200" />
                  <div className="relative z-10">
                    <p className="text-xs text-gray-500 mb-1">{topic.subtitle}</p>
                    <p className="text-sm font-bold text-gray-900">{topic.title}</p>
                  </div>
                </div>
              ))}
            </div>
            <button className="relative w-full p-4 text-sm text-gray-900 font-medium text-left">
              <div className="absolute inset-0 bg-gray-200 opacity-0 hover:opacity-100 transition-opacity duration-200" />
              <span className="relative z-10">Show more</span>
            </button>
          </div>

          {/* Who to Follow */}
          <div className="bg-gray-50 rounded-2xl overflow-hidden">
            <div className="p-4">
              <h2 className="text-xl font-bold text-gray-900">Who to follow</h2>
            </div>
            <div>
              {whoToFollow.map((user, idx) => (
                <div key={idx} className="relative px-4 py-3">
                  <div className="absolute inset-0 bg-gray-200 opacity-0 hover:opacity-100 transition-opacity duration-200" />
                  <div className="relative z-10 flex items-center justify-between">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <div className="w-10 h-10 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                        <User className="w-5 h-5 text-gray-600" strokeWidth={2} />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-bold text-gray-900 truncate">{user.name}</p>
                        <p className="text-sm text-gray-500 truncate">{user.handle}</p>
                      </div>
                    </div>
                    <button className="px-4 py-1.5 bg-black hover:bg-gray-800 text-white text-sm font-bold rounded-full transition-colors duration-200 flex-shrink-0">
                      Follow
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <button className="relative w-full p-4 text-sm text-gray-900 font-medium text-left">
              <div className="absolute inset-0 bg-gray-200 opacity-0 hover:opacity-100 transition-opacity duration-200" />
              <span className="relative z-10">Show more</span>
            </button>
          </div>
        </aside>
      </div>

      {/* Mobile Bottom Navigation */}
      <nav className="md:hidden fixed bottom-0 inset-x-0 bg-white border-t border-gray-200 z-40">
        <div className="flex items-center justify-around px-2 py-2">
          {navItems.slice(0, 5).map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link key={item.path} to={item.path} className="flex-1">
                <div className={`flex flex-col items-center gap-1 py-2 ${isActive ? "text-gray-900" : "text-gray-500"}`}>
                  <item.icon className="w-6 h-6" strokeWidth={isActive ? 2.5 : 2} />
                  <span className="text-[10px] font-medium">{item.label}</span>
                </div>
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
};

export default Layout;
