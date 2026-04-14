'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { AuthForm } from '@/components/auth-form';
import { ChatRoom } from '@/components/chat-room';
import { Button } from '@/components/ui/button';
import { ModeToggle } from '@/components/ui/mode-toggle';
import { User, LogOut, MessageSquare } from 'lucide-react';
import { motion } from 'framer-motion';
import { BackgroundElements } from '@/components/background-elements';

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
    <main className="min-h-screen relative overflow-x-hidden">
      <BackgroundElements />
      <nav className="border-b bg-background/20 backdrop-blur-xl sticky top-0 z-50 border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-primary/20 p-2 rounded-xl backdrop-blur-md ring-1 ring-white/20">
              <MessageSquare className="w-5 h-5 text-primary" />
            </div>
            <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-foreground to-foreground/70">Haven</h1>
          </div>
          
          <div className="flex items-center gap-4">
            <ModeToggle />
            {session && (
              <div className="flex items-center gap-4">
                <div className="hidden md:flex flex-col items-end">
                  <span className="text-sm font-medium">{session.user.email}</span>
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Authenticated</span>
                </div>
                <Button variant="ghost" size="icon" onClick={handleSignOut} className="rounded-xl hover:bg-destructive/10 hover:text-destructive transition-colors">
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
          <div className="flex flex-col items-center justify-center min-h-[80vh] space-y-12 py-12">
            <div className="text-center space-y-8 max-w-4xl px-4 relative">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="inline-flex items-center gap-2 px-4 py-1.5 mb-4 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-bold uppercase tracking-widest backdrop-blur-sm"
              >
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                Production-Grade AI Moderation
              </motion.div>
              <h2 className="text-6xl font-extrabold tracking-tight sm:text-8xl bg-clip-text text-transparent bg-gradient-to-b from-foreground to-foreground/50 leading-[1] py-2">
                A sanctuary for <br />
                <span className="text-primary relative inline-block">
                  Your Conversations
                  <div className="absolute -bottom-2 left-0 w-full h-2 bg-primary/20 rounded-full blur-md" />
                </span>
              </h2>
              <p className="text-muted-foreground text-xl max-w-2xl mx-auto leading-relaxed font-medium">
                Haven: A place of safety or refuge from the rest of the toxic internet.
              </p>
              
              <div className="relative pt-8">
                <div className="absolute inset-0 bg-primary/5 blur-3xl rounded-full" />
                <AuthForm />
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
