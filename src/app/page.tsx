'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { AuthForm } from '@/components/auth-form';
import { ChatRoom } from '@/components/chat-room';
import { Button } from '@/components/ui/button';
import { ModeToggle } from '@/components/ui/mode-toggle';
import { User, LogOut, MessageSquare } from 'lucide-react';
import { motion } from 'framer-motion';

import { Session } from '@supabase/supabase-js';

export default function Home() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then((res: { data: { session: Session | null } }) => {
      setSession(res.data.session);
      setLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event: string, currentSession: Session | null) => {
      setSession(currentSession);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <nav className="border-b bg-background/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-primary p-2 rounded-lg">
              <MessageSquare className="w-5 h-5 text-primary-foreground" />
            </div>
            <h1 className="text-xl font-bold tracking-tight">Bloom Chat</h1>
          </div>
          
          <div className="flex items-center gap-4">
            <ModeToggle />
            {session && (
              <div className="flex items-center gap-4">
                <div className="hidden md:flex flex-col items-end">
                  <span className="text-sm font-medium">{session.user.email}</span>
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Authenticated</span>
                </div>
                <Button variant="ghost" size="icon" onClick={handleSignOut}>
                  <LogOut className="w-5 h-5" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </nav>

      <div className="container mx-auto py-8 px-4">
        {session ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <ChatRoom user={session.user} />
          </motion.div>
        ) : (
          <div className="flex flex-col items-center justify-center min-h-[70vh] space-y-12">
            <div className="text-center space-y-6 max-w-3xl px-4">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="inline-block px-4 py-1.5 mb-4 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-widest"
              >
                Production-Grade AI Moderation
              </motion.div>
              <h2 className="text-5xl font-extrabold tracking-tight sm:text-7xl bg-clip-text text-transparent bg-gradient-to-b from-foreground to-foreground/70 leading-[1.1]">
                Real-time chat with <br />
                <span className="text-primary relative inline-block">
                  AI Moderation
                  <div className="absolute -bottom-2 left-0 w-full h-1 bg-primary/20 rounded-full blur-sm" />
                </span>
              </h2>
              <p className="text-muted-foreground text-xl max-w-xl mx-auto leading-relaxed">
                Experience the next generation of community platforms. 
                Sleek design meets Gemini-powered safety.
              </p>
            </div>
            <AuthForm />
          </div>
        )}
      </div>
    </main>
  );
}
