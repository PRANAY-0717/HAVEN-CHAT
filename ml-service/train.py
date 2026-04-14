import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Updated training data to handle false positives like 'okok'
raw_data = [
    # --- NON-TOXIC (Label 0) ---
    ('I love this project, it is amazing!', 0),
    ('You are a great person.', 0),
    ('Have a nice day!', 0),
    ('This is a helpful tool.', 0),
    ('This is the best thing ever!', 0),
    ('Hello there, how are you?', 0),
    ('I disagree with your opinion but respect it.', 0),
    ('You are doing a fantastic job!', 0),
    ('You are the best.', 0),
    ('You are amazing.', 0),
    ('I really like you.', 0),
    ('This is wonderful.', 0),
    ('You are so kind.', 0),
    ('Thank you for your help.', 0),
    ('Have a beautiful day.', 0),
    ('I appreciate your hard work.', 0),
    ('You are doing great.', 0),
    ('I love you.', 0),
    ('This is great news.', 0),
    ('You are a hero.', 0),
    ('Keep up the good work.', 0),
    ('Can you help me with this?', 0),
    ('What a wonderful afternoon.', 0),
    ('The weather is lovely.', 0),
    ('I am so happy for you.', 0),
    ('Congratulations on your success!', 0),
    ('That is a very interesting point.', 0),
    ('I would like to learn more about this.', 0),
    ('Let us work together on this.', 0),
    ('ok', 0),
    ('okok', 0),
    ('yes', 0),
    ('no', 0),
    ('hello', 0),
    ('hi', 0),
    ('hey', 0),
    ('lol', 0),
    ('wow', 0),
    ('good', 0),
    ('nice', 0),
    ('cool', 0),
    ('see you', 0),
    ('bye', 0),
    ('thanks', 0),
    ('thank you', 0),
    ('ok thanks', 0),
    ('yes please', 0),
    ('no thanks', 0),
    ('just ok', 0),
    ('it is ok', 0),
    ('ok let us go', 0),
    ('ok fine', 0),
    ('okok got it', 0),
    ('ok wait', 0),
    ('how are you', 0),
    ('what is up', 0),
    ('nothing much', 0),
    ('all good', 0),
    ('not bad', 0),

    # --- TOXIC (Label 1) ---
    ('I hate you so much, you are terrible.', 1),
    ('Go away, nobody likes you.', 1),
    ('You are an idiot and a loser.', 1),
    ('Shut up, you are so annoying.', 1),
    ('You suck and your code is garbage.', 1),
    ('Get lost, you piece of trash.', 1),
    ('Die please, you are worthless.', 1),
    ('Kill yourself.', 1),
    ('I will hurt you.', 1),
    ('You should not be alive.', 1),
    ('Stupid moron.', 1),
    ('Fuck off.', 1),
    ('You are a complete failure.', 1),
    ('You are a disgusting pig.', 1),
    ('I hope you fail in life.', 1),
    ('You are a pathetic excuse for a human.', 1),
    ('Nobody wants you here.', 1),
    ('Your existence is a mistake.', 1),
    ('You are so incredibly stupid.', 1),
    ('Go rot in hell.', 1),
    ('You are a useless waste of space.', 1),
    ('I hate everything about you.', 1),
    ('You make me sick.', 1),
    ('You are the worst person I have ever met.', 1),
    ('Oh great, another brilliant idea from you.', 1),
    ('You must be so proud of your failure.', 1),
    ('I love how you think you know everything.', 1),
    ('Wow, you are so smart (not).', 1),
    ('Maybe try using your brain for once?', 1),
    ('Your ignorance is truly impressive.', 1),
    ('Bitch please.', 1),
    ('You are a total asshole.', 1),
    ('What a dumbass.', 1),
    ('Stop being such a prick.', 1),
    ('You are a coward and a liar.', 1),
    ('Don\'t be such a jerk.', 1),
    ('You are acting like a fool.', 1)
]

df = pd.DataFrame(raw_data, columns=['text', 'is_toxic'])

def train_model():
    print(f"Training model with {len(df)} examples...")
    X = df['text']
    y = df['is_toxic']

    tfidf = TfidfVectorizer(
        stop_words='english', 
        lowercase=True, 
        ngram_range=(1, 3), 
        analyzer='word',
        max_features=5000
    )
    X_tfidf = tfidf.fit_transform(X)

    model = LogisticRegression(
        class_weight='balanced', 
        C=0.5, 
        solver='liblinear',
        max_iter=1000
    )
    model.fit(X_tfidf, y)

    if not os.path.exists('models'):
        os.makedirs('models')
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    print("Model and vectorizer saved to models/ folder.")

if __name__ == "__main__":
    train_model()
