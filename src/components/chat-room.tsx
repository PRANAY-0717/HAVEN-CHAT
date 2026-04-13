'use client';

import { useEffect, useState, useRef } from 'react';
import { supabase } from '@/lib/supabase';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { motion, AnimatePresence } from 'framer-motion';
import { useDebounce } from '@/hooks/use-debounce';
import { RealtimePostgresInsertPayload } from '@supabase/supabase-js';
import { ChatDashboard } from '@/components/chat-dashboard';
import { AIInsights } from '@/components/ai-insights';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Send, ShieldAlert, ShieldCheck, Loader2, BarChart3, Brain, Settings2, Zap, Database, MessageSquare } from 'lucide-react';

interface Profile {
  username: string;
  avatar_url: string;
}

interface Message {
  id: string;
  user_id: string;
  content: string;
  is_toxic: boolean;
  toxicity_score: number;
  created_at: string;
  profiles?: Profile;
}

interface ChatRoomProps {
  user: any;
}

export function ChatRoom({ user }: ChatRoomProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [toxicityScore, setToxicityScore] = useState(0);
  const [moderationMode, setModerationMode] = useState<'local' | 'high_accuracy'>('local');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const debouncedMessage = useDebounce(newMessage, 500);

  useEffect(() => {
    const checkToxicity = async () => {
      if (debouncedMessage.trim().length > 3) {
        try {
          const mlApiUrl = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000';
          const res = await fetch(`${mlApiUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: debouncedMessage, mode: moderationMode }),
          });
          if (res.ok) {
            const { toxicity_score } = await res.json();
            setToxicityScore(toxicity_score);
          }
        } catch (err) {
          console.error('Toxicity check error:', err);
        }
      } else {
        setToxicityScore(0);
      }
    };
    checkToxicity();
  }, [debouncedMessage, moderationMode]);

  useEffect(() => {
    // Ensure profile exists for the current user
    const ensureProfile = async () => {
      if (!user) return;

      const { data: profile, error } = await supabase
        .from('profiles')
        .select('id')
        .eq('id', user.id)
        .single();

      if (error && error.code === 'PGRST116') {
        // Profile doesn't exist, create it
        console.log('Creating missing profile for user:', user.id);
        const { error: insertError } = await supabase
          .from('profiles')
          .insert({
            id: user.id,
            username: user.email?.split('@')[0] || 'user_' + Math.random().toString(36).substring(7),
            avatar_url: user.user_metadata?.avatar_url || '',
          });

        if (insertError) {
          console.error('Error creating profile:', insertError);
          toast.error('Failed to initialize your profile. Chat may be restricted.');
        }
      }
    };

    ensureProfile();
  }, [user]);

  useEffect(() => {
    // Fetch initial messages
    const fetchMessages = async () => {
      const { data, error } = await supabase
        .from('messages')
        .select('*, profiles(username, avatar_url)')
        .order('created_at', { ascending: true })
        .limit(50);

      if (error) {
        toast.error('Failed to fetch messages');
      } else {
        setMessages(data || []);
      }
    };

    fetchMessages();

    // Subscribe to real-time changes
    console.log('Setting up real-time subscription...');
    const channel = supabase
      .channel('db-messages')
      .on(
        'postgres_changes',
        { 
          event: 'INSERT', 
          schema: 'public', 
          table: 'messages' 
        },
        async (payload: RealtimePostgresInsertPayload<Message>) => {
          console.log('Real-time message received:', payload.new);
          
          // Fetch profile for the new message
          const { data: profileData, error: profileError } = await supabase
            .from('profiles')
            .select('username, avatar_url')
            .eq('id', payload.new.user_id)
            .single();

          if (profileError) {
            console.error('Error fetching profile for real-time message:', profileError);
          }

          const newMessage = {
            ...payload.new,
            profiles: profileData as Profile,
          } as Message;

          setMessages((prev) => {
            // Remove any optimistic message with the same content and user that was just sent
            const filtered = prev.filter(m => 
              !(m.id.startsWith('optimistic-') && 
                m.content === newMessage.content && 
                m.user_id === newMessage.user_id)
            );
            
            // Check if this message already exists (to be safe)
            if (filtered.some(m => m.id === newMessage.id)) return filtered;
            
            console.log('Adding new message to state:', newMessage);
            return [...filtered, newMessage];
          });
          scrollToBottom();
        }
      )
      .subscribe((status: string, err?: any) => {
        console.log('Real-time subscription status:', status);
        if (err) console.error('Subscription error object:', err);
        
        if (status === 'CHANNEL_ERROR') {
          console.error('Real-time channel error occurred');
          toast.error('Real-time chat is disconnected. Try reloading.');
        }
      });

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  useEffect(() => {
    // Small delay to ensure the DOM has finished updating
    const timeoutId = setTimeout(() => {
      scrollToBottom();
    }, 100);
    return () => clearTimeout(timeoutId);
  }, [messages]);

  const scrollToBottom = (force = false) => {
    if (messagesEndRef.current) {
      // Only auto-scroll if user is already near the bottom or if forced
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim() || loading) return;

    setLoading(true);

    try {
      const mlApiUrl = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000';
      // 1. Toxicity Check (Intercept)
      const res = await fetch(`${mlApiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newMessage, mode: moderationMode }),
      });

      if (!res.ok) throw new Error('Failed to check toxicity');

      const { toxicity_score, is_toxic } = await res.json();

      let finalContent = newMessage;
      if (is_toxic) {
        finalContent = '[Message removed by Auto-Moderator]';
        toast.warning('⚠️ Your message violated community guidelines', {
          description: 'Keep the conversation respectful.',
          duration: 5000,
        });
      }

      // 2. Store in Supabase
      const { error } = await supabase.from('messages').insert({
        user_id: user.id,
        content: finalContent,
        is_toxic,
        toxicity_score,
      });

      if (error) {
        if (error.code === '23503') {
          toast.error('Profile not found. Re-initializing...', {
            description: 'Please wait a moment and try again.',
          });
          // Trigger profile check again
          const { error: profileError } = await supabase
            .from('profiles')
            .upsert({
              id: user.id,
              username: user.email?.split('@')[0] || 'user_' + Math.random().toString(36).substring(7),
              avatar_url: user.user_metadata?.avatar_url || '',
            });
          if (profileError) console.error('Failed to upsert profile:', profileError);
          setLoading(false);
          return;
        } else {
          throw error;
        }
      }

      // Optimistic update for the sender (FAANG-level speed)
      const optimisticMsg: Message = {
        id: 'optimistic-' + Math.random().toString(), // Use a prefix
        user_id: user.id,
        content: finalContent,
        is_toxic,
        toxicity_score,
        created_at: new Date().toISOString(),
        profiles: { 
          username: user.email?.split('@')[0] || 'demo', 
          avatar_url: user.user_metadata?.avatar_url || '' 
        },
      };
      
      setMessages((prev) => [...prev, optimisticMsg]);

      setNewMessage('');
      setToxicityScore(0);
    } catch (error: any) {
      toast.error(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTyping = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNewMessage(e.target.value);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] max-w-4xl mx-auto p-4 space-y-4">
      <Card className="flex-1 flex flex-col bg-background/50 backdrop-blur-md border-muted-foreground/20 shadow-xl overflow-hidden min-h-0">
        <CardHeader className="border-b bg-muted/30 py-4 flex flex-row items-center justify-between shrink-0">
          <div>
            <CardTitle className="text-xl font-bold flex items-center gap-2">
              <span className="text-primary">Bloom</span> Chat
              <Badge variant="outline" className="ml-2 bg-green-500/10 text-green-500 border-green-500/20">
                Live
              </Badge>
            </CardTitle>
          </div>
          <div className="flex items-center gap-4">
            <Sheet>
              <SheetTrigger
                render={
                  <Button variant="ghost" size="icon" className="hover:bg-primary/10">
                    <BarChart3 className="w-5 h-5 text-muted-foreground" />
                  </Button>
                }
              />
              <SheetContent className="w-full sm:max-w-md">
                <SheetHeader className="mb-6">
                  <SheetTitle className="text-xl">Chat Analytics</SheetTitle>
                  <SheetDescription>Real-time community insights and behavior monitoring.</SheetDescription>
                </SheetHeader>
                <ChatDashboard messages={messages} />
              </SheetContent>
            </Sheet>

            <Sheet>
              <SheetTrigger
                render={
                  <Button variant="ghost" size="icon" className="hover:bg-primary/10">
                    <Brain className="w-5 h-5 text-muted-foreground" />
                  </Button>
                }
              />
              <SheetContent className="w-full sm:max-w-md">
                <SheetHeader className="mb-6">
                  <SheetTitle className="text-xl">AI Behavior Insights</SheetTitle>
                  <SheetDescription>Personalized analysis of your communication style.</SheetDescription>
                </SheetHeader>
                <AIInsights userMessages={messages.filter((m) => m.user_id === user.id)} />
              </SheetContent>
            </Sheet>

            <DropdownMenu>
              <DropdownMenuTrigger
                render={
                  <Button variant="ghost" size="icon" className="hover:bg-primary/10">
                    <Settings2 className="w-5 h-5 text-muted-foreground" />
                  </Button>
                }
              />
              <DropdownMenuContent align="end" className="w-56">
                <div className="p-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Moderation Mode
                </div>
                <DropdownMenuItem 
                  onClick={() => setModerationMode('local')}
                  className="flex items-center gap-2"
                >
                  <Database className={`w-4 h-4 ${moderationMode === 'local' ? 'text-primary' : ''}`} />
                  <div className="flex flex-col">
                    <span>Local (Fast)</span>
                    <span className="text-[10px] text-muted-foreground">Logistic Regression</span>
                  </div>
                  {moderationMode === 'local' && <div className="ml-auto w-2 h-2 rounded-full bg-primary" />}
                </DropdownMenuItem>
                <DropdownMenuItem 
                  onClick={() => setModerationMode('high_accuracy')}
                  className="flex items-center gap-2"
                >
                  <Zap className={`w-4 h-4 ${moderationMode === 'high_accuracy' ? 'text-primary' : ''}`} />
                  <div className="flex flex-col">
                    <span>High Accuracy</span>
                    <span className="text-[10px] text-muted-foreground">Gemini 1.5 Flash</span>
                  </div>
                  {moderationMode === 'high_accuracy' && <div className="ml-auto w-2 h-2 rounded-full bg-primary" />}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <div className="flex items-center gap-2 border-l pl-4 ml-2">
              {toxicityScore > 0.5 ? (
                <ShieldAlert className="w-5 h-5 text-red-500 animate-pulse" />
              ) : (
                <ShieldCheck className="w-5 h-5 text-green-500" />
              )}
              <div className="w-24 h-2 bg-muted rounded-full overflow-hidden hidden sm:block">
                <motion.div
                  className="h-full bg-red-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${toxicityScore * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </div>
          </div>
        </CardHeader>

        <div className="flex-1 overflow-y-auto p-6 space-y-8 custom-scrollbar bg-gradient-to-b from-background/0 to-muted/20">
          <div className="flex flex-col space-y-6 min-h-full">
            <div className="flex flex-col items-center justify-center py-8 opacity-40">
              <div className="p-3 rounded-full bg-primary/10 mb-2">
                <MessageSquare className="w-6 h-6 text-primary" />
              </div>
              <p className="text-xs font-medium uppercase tracking-widest">End-to-End Encrypted</p>
            </div>
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.2 }}
                  className={`flex ${msg.user_id === user.id ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`flex gap-3 max-w-[85%] ${msg.user_id === user.id ? 'flex-row-reverse' : 'flex-row'}`}>
                    <Avatar className="w-10 h-10 mt-1 ring-2 ring-background shadow-sm shrink-0">
                      <AvatarImage src={msg.profiles?.avatar_url} />
                      <AvatarFallback className="text-xs uppercase bg-gradient-to-br from-primary to-primary/60 text-primary-foreground">
                        {msg.profiles?.username?.substring(0, 2) || user.email?.substring(0, 2)}
                      </AvatarFallback>
                    </Avatar>
                    <div className={`flex flex-col ${msg.user_id === user.id ? 'items-end' : 'items-start'}`}>
                      <div className={`flex items-center gap-2 mb-1.5 px-1 ${msg.user_id === user.id ? 'flex-row-reverse' : 'flex-row'}`}>
                        <span className="text-xs font-semibold text-foreground/80">
                          {msg.profiles?.username || 'Anonymous'}
                        </span>
                        <span className="text-[10px] text-muted-foreground/50 font-medium">
                          {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                      <div
                        className={`group relative px-4 py-2.5 rounded-2xl shadow-sm text-sm transition-all duration-200 hover:shadow-md ${
                          msg.is_toxic
                            ? 'bg-destructive/10 border border-destructive/20 text-destructive italic backdrop-blur-sm'
                            : msg.user_id === user.id
                            ? 'bg-primary text-primary-foreground rounded-tr-none'
                            : 'bg-muted/50 text-foreground border border-white/5 backdrop-blur-sm rounded-tl-none'
                        }`}
                      >
                        {msg.content}
                        {!msg.is_toxic && (
                          <div className={`absolute -bottom-5 opacity-0 group-hover:opacity-100 transition-opacity text-[9px] text-muted-foreground whitespace-nowrap ${msg.user_id === user.id ? 'right-0' : 'left-0'}`}>
                            {msg.id.startsWith('optimistic-') ? 'Sending...' : 'Delivered'}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        </div>

        <CardFooter className="p-4 border-t bg-background/40 backdrop-blur-sm shrink-0">
          <form onSubmit={handleSendMessage} className="flex w-full gap-3 relative max-w-3xl mx-auto">
            <div className="flex-1 relative group">
              <Input
                placeholder="Type your message..."
                value={newMessage}
                onChange={handleTyping}
                className="w-full rounded-2xl bg-muted/30 border-white/5 focus-visible:ring-primary h-12 px-6 pr-12 shadow-inner transition-all group-hover:bg-muted/50"
                disabled={loading}
              />
              <div className="absolute right-4 top-1/2 -translate-y-1/2">
                <div className={`w-1.5 h-1.5 rounded-full transition-colors duration-500 ${toxicityScore > 0.7 ? 'bg-destructive animate-pulse' : toxicityScore > 0.3 ? 'bg-yellow-500' : 'bg-green-500/40'}`} />
              </div>
            </div>
            <Button
              type="submit"
              size="icon"
              className="rounded-2xl w-12 h-12 shrink-0 bg-primary hover:scale-105 active:scale-95 transition-all shadow-lg shadow-primary/20"
              disabled={loading || !newMessage.trim()}
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  );
}
