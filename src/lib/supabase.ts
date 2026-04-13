import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

const isMock = !supabaseUrl || !supabaseAnonKey || supabaseUrl === 'your-supabase-url' || supabaseAnonKey === 'your-supabase-anon-key';

// Mock Supabase client for demonstration purposes when real credentials are not provided
class MockSupabaseClient {
  auth = {
    getSession: async () => ({ data: { session: { user: { id: 'demo-user', email: 'demo@example.com' } } }, error: null }),
    onAuthStateChange: (callback: any) => ({ data: { subscription: { unsubscribe: () => {} } } }),
    signInWithPassword: async () => ({ data: { user: { id: 'demo-user' } }, error: null }),
    signUp: async () => ({ data: { user: { id: 'demo-user' } }, error: null }),
    signOut: async () => ({ error: null }),
    signInWithOAuth: async () => ({ error: null }),
  };

  from(table: string) {
    return {
      select: () => ({
        order: () => ({
          limit: () => Promise.resolve({
            data: table === 'messages' ? [
              {
                id: '1',
                user_id: 'bot-1',
                content: 'Welcome to Bloom Chat! Try typing something.',
                is_toxic: false,
                toxicity_score: 0.01,
                created_at: new Date().toISOString(),
                profiles: { username: 'Bloom Bot', avatar_url: '' }
              }
            ] : [],
            error: null
          })
        }),
        eq: () => ({
          single: () => Promise.resolve({ data: { username: 'Demo User', avatar_url: '' }, error: null })
        })
      }),
      insert: (data: any) => {
        console.log('Mock Insert:', data);
        return Promise.resolve({ error: null });
      }
    };
  }

  channel(name: string) {
    return {
      on: (event: string, filter: any, callback: any) => ({
        subscribe: () => ({ unsubscribe: () => {} })
      }),
      subscribe: () => ({ unsubscribe: () => {} })
    } as any;
  }

  removeChannel(channel: any) {}
}

export const supabase = isMock 
  ? (new MockSupabaseClient() as any) 
  : createClient(supabaseUrl, supabaseAnonKey);
