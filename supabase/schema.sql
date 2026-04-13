-- Create profiles table
create table profiles (
  id uuid references auth.users on delete cascade primary key,
  username text unique,
  avatar_url text,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create messages table
create table messages (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references profiles(id) on delete cascade not null,
  content text not null,
  is_toxic boolean default false,
  toxicity_score float default 0.0,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable RLS
alter table profiles enable row level security;
alter table messages enable row level security;

-- Profiles policies
create policy "Public profiles are viewable by everyone" on profiles
  for select using (true);

create policy "Users can insert their own profile" on profiles
  for insert with check (auth.uid() = id);

create policy "Users can update own profile" on profiles
  for update using (auth.uid() = id);

-- Messages policies
create policy "Messages are viewable by everyone" on messages
  for select using (true);

create policy "Authenticated users can insert messages" on messages
  for insert with check (auth.role() = 'authenticated');

-- Realtime setup
alter publication supabase_realtime add table messages;
alter table messages replica identity full;

-- Auto-profile trigger
create function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, username, avatar_url)
  values (new.id, split_part(new.email, '@', 1), new.raw_user_meta_data->>'avatar_url');
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
