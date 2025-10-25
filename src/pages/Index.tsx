import Layout from "@/components/Layout";
import Post from "@/components/Post";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const Index = () => {
  const posts = [
    {
      username: "dotstellaris",
      verified: true,
      time: "3h",
      content: "OMG 🎉 celebrating 🥳 over 4000 followers today! Thank you! Enjoy this augmented reality real time puppet I made. You can try it now were below in the thread",
      replies: 2482,
      likes: 378,
      avatar: "D"
    },
    {
      username: "arcastic_us",
      verified: true,
      time: "12h",
      content: "",
      image: "https://images.unsplash.com/photo-1635805737707-575885ab0b9f?w=800&q=80",
      replies: 540,
      likes: 12000,
      avatar: "A"
    },
    {
      username: "natsdaily",
      verified: true,
      time: "1d",
      content: `This place is called "Sealand". It is a KM off the coast of the UK. It was a military structure that was later abandoned.

So someone named Roy Bates decided to take it over and turn it into his own country in International Waters.`,
      replies: 372,
      likes: 3600,
      avatar: "N"
    }
  ];

  return (
    <Layout>
      {/* Post input */}
      <div className="border-b border-border p-6">
        <div className="flex gap-3">
          <Avatar className="w-10 h-10">
            <AvatarImage src="" />
            <AvatarFallback className="bg-secondary text-foreground">R</AvatarFallback>
          </Avatar>
          <div className="flex-1">
            <textarea
              placeholder="Start a thread..."
              className="w-full bg-transparent text-foreground placeholder:text-muted-foreground resize-none outline-none text-[15px]"
              rows={2}
            />
            <div className="flex items-center justify-between mt-2">
              <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground h-auto p-0">
                📎
              </Button>
              <Button size="sm" className="bg-accent text-accent-foreground hover:bg-accent/90 rounded-full px-6">
                Post
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Posts Feed */}
      <div>
        {posts.map((post, index) => (
          <Post key={index} {...post} />
        ))}
      </div>
    </Layout>
  );
};

export default Index;
