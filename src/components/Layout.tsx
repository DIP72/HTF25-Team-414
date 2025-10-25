import { ReactNode } from "react";
import { Home, Search, Plus, Heart, User, Menu } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Top Logo */}
      <div className="fixed top-0 left-0 right-0 h-16 flex items-center justify-center border-b border-border bg-background z-50">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" className="text-foreground">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
          <path d="M12 6C8.7 6 6 8.7 6 12C6 15.3 8.7 18 12 18" stroke="currentColor" strokeWidth="2"/>
        </svg>
      </div>

      <div className="pt-16 flex max-w-[1400px] mx-auto">
        {/* Left Sidebar */}
        <aside className="fixed left-0 w-64 h-[calc(100vh-4rem)] border-r border-border p-6 hidden lg:block">
          <nav className="space-y-6">
            <Button variant="ghost" className="w-full justify-start gap-4 text-foreground hover:bg-secondary">
              <Home className="w-6 h-6" />
              <span className="text-base">Home</span>
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-4 text-muted-foreground hover:bg-secondary hover:text-foreground">
              <Search className="w-6 h-6" />
              <span className="text-base">Search</span>
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-4 text-muted-foreground hover:bg-secondary hover:text-foreground">
              <Plus className="w-6 h-6" />
              <span className="text-base">Create</span>
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-4 text-muted-foreground hover:bg-secondary hover:text-foreground">
              <Heart className="w-6 h-6" />
              <span className="text-base">Activity</span>
            </Button>
            <Button variant="ghost" className="w-full justify-start gap-4 text-muted-foreground hover:bg-secondary hover:text-foreground">
              <User className="w-6 h-6" />
              <span className="text-base">Profile</span>
            </Button>
          </nav>

          <div className="absolute bottom-6">
            <Button variant="ghost" className="w-full justify-start gap-4 text-muted-foreground hover:bg-secondary hover:text-foreground">
              <Menu className="w-6 h-6" />
              <span className="text-base">More</span>
            </Button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 lg:ml-64 lg:mr-80 border-r border-border">
          {children}
        </main>

        {/* Right Sidebar */}
        <aside className="fixed right-0 w-80 h-[calc(100vh-4rem)] p-6 hidden lg:block overflow-y-auto">
          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-muted-foreground text-sm font-medium">Suggested for you</h2>
                <Button variant="ghost" size="sm" className="text-foreground text-sm h-auto p-0 hover:bg-transparent">
                  See All
                </Button>
              </div>

              <div className="space-y-4">
                {[
                  { name: "mkbhd", subtitle: "Followed you", avatar: "M" },
                  { name: "main9", subtitle: "Suggested for you", avatar: "M" },
                  { name: "maddixryan", subtitle: "Suggested for you", avatar: "M" },
                  { name: "isaymakr", subtitle: "Suggested for you", avatar: "I" },
                  { name: "jhanith", subtitle: "Suggested for you", avatar: "J" },
                ].map((user, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Avatar className="w-9 h-9">
                        <AvatarImage src="" />
                        <AvatarFallback className="bg-secondary text-foreground text-sm">{user.avatar}</AvatarFallback>
                      </Avatar>
                      <div>
                        <p className="text-sm font-medium text-foreground">{user.name}</p>
                        <p className="text-xs text-muted-foreground">{user.subtitle}</p>
                      </div>
                    </div>
                    <Button size="sm" className="bg-primary text-primary-foreground hover:bg-primary/90 h-8 px-4 rounded-lg">
                      Follow
                    </Button>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <p className="text-sm text-muted-foreground mb-3">Download mobile app</p>
              <div className="bg-card border border-border rounded-lg p-4 flex items-center justify-center">
                <div className="w-32 h-32 bg-foreground rounded-lg"></div>
              </div>
            </div>

            <div className="text-xs text-muted-foreground space-y-2">
              <div className="flex flex-wrap gap-2">
                <span>About</span>
                <span>•</span>
                <span>Help</span>
                <span>•</span>
                <span>Press</span>
                <span>•</span>
                <span>API</span>
                <span>•</span>
                <span>Privacy</span>
                <span>•</span>
                <span>Terms</span>
              </div>
              <div className="flex flex-wrap gap-2">
                <span>Locations</span>
                <span>•</span>
                <span>Language</span>
                <span>•</span>
                <span>Meta Verified</span>
              </div>
              <p className="pt-2">© 2024 Meta</p>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default Layout;
