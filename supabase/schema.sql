-- Create profiles table
create table if not exists profiles (
  id uuid references auth.users on delete cascade primary key,
  username text unique,
  avatar_url text,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create messages table
create table if not exists messages (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references profiles(id) on delete cascade not null,
  content text not null,
  is_toxic boolean default false,
  toxicity_score float default 0.0,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create saved_messages table
create table if not exists saved_messages (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references profiles(id) on delete cascade not null,
  message_id uuid references messages(id) on delete cascade not null,
  saved_at timestamp with time zone default timezone('utc'::text, now()) not null,
  unique(user_id, message_id)
);

-- Enable RLS
alter table profiles enable row level security;
alter table messages enable row level security;
alter table saved_messages enable row level security;

-- Profiles policies
do $$
begin
  drop policy if exists "Public profiles are viewable by everyone" on profiles;
  create policy "Public profiles are viewable by everyone" on profiles for select using (true);
  
  drop policy if exists "Users can insert their own profile" on profiles;
  create policy "Users can insert their own profile" on profiles for insert with check (auth.uid() = id);
  
  drop policy if exists "Users can update own profile" on profiles;
  create policy "Users can update own profile" on profiles for update using (auth.uid() = id);
end $$;

-- Messages policies
do $$
begin
  drop policy if exists "Messages are viewable by everyone" on messages;
  create policy "Messages are viewable by everyone" on messages for select using (true);
  
  drop policy if exists "Authenticated users can insert messages" on messages;
  create policy "Authenticated users can insert messages" on messages for insert with check (auth.role() = 'authenticated');
  
  drop policy if exists "Authenticated users can delete messages" on messages;
  create policy "Authenticated users can delete messages" on messages for delete using (auth.role() = 'authenticated');
end $$;

-- Saved Messages policies
do $$
begin
  drop policy if exists "Users can view their own saved messages" on saved_messages;
  create policy "Users can view their own saved messages" on saved_messages for select using (auth.uid() = user_id);
  
  drop policy if exists "Users can save messages" on saved_messages;
  create policy "Users can save messages" on saved_messages for insert with check (auth.uid() = user_id);
  
  drop policy if exists "Users can unsave their own messages" on saved_messages;
  create policy "Users can unsave their own messages" on saved_messages for delete using (auth.uid() = user_id);
end $$;

-- Realtime setup
do $$
begin
  if not exists (select 1 from pg_publication_tables where pubname = 'supabase_realtime' and tablename = 'messages') then
    alter publication supabase_realtime add table messages;
  end if;
end $$;
alter table messages replica identity full;

-- Function to cleanup old messages (every 20 mins)
create or replace function public.cleanup_old_messages()
returns void
language plpgsql
security definer
as $$
begin
  delete from public.messages
  where created_at < now() - interval '20 minutes'
  and id not in (select message_id from public.saved_messages);
end;
$$;

-- Auto-profile trigger
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, username, avatar_url)
  values (new.id, split_part(new.email, '@', 1), new.raw_user_meta_data->>'avatar_url')
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
