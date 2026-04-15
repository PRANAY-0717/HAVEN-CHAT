import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Refined dataset with insights from research on subtle toxicity and common chat patterns
raw_data = [
    # --- NON-TOXIC: Standard Positive/Neutral (Label 0) ---
    ('I love this project, it is amazing!', 0),
    ('You are a great person.', 0),
    ('Have a nice day!', 0),
    ('This is a helpful tool.', 0),
    ('This is the best thing ever!', 0),
    ('Hello there, how are you?', 0),
    ('I disagree with your opinion but respect it.', 0),
    ('You are doing a fantastic job!', 0),
    ('Thank you for your help.', 0),
    ('I appreciate your hard work.', 0),
    ('Can you help me with this?', 0),
    ('Let us work together on this.', 0),
    ('Exactly', 0), ('correct', 0), ('true', 0), ('i agree', 0),
    ('ok', 0), ('yes', 0), ('no', 0), ('hello', 0), ('thanks', 0),
    
    # --- NON-TOXIC: Adversarial / Slang / Gaming (Label 0) ---
    # (Crucial for teaching TF-IDF that "aggressive" words aren't always toxic)
    ('kill the process', 0), 
    ('crash the server', 0),
    ('execute the program', 0),
    ('dump the memory', 0),
    ('headshot', 0),
    ('gg wp', 0),
    ('destroy the enemy base', 0),
    ('this new feature is sick', 0),
    ('that was a badass presentation', 0),
    ('f**k yeah, we did it', 0), # Celebratory
    ('holy cow', 0), 
    ('my bad', 0), 
    ('no worries', 0), 
    ('I am going to kill this exam', 0),
    ('this pizza is the bomb', 0),
    ('not bad at all', 0),

    # --- NON-TOXIC: Hinglish / Multilingual (Label 0) ---
    ('kya haal hai bhai', 0),
    ('sahi hai yaar', 0),
    ('tu toh hero hai', 0),
    ('shabash', 0),
    ('mast lag raha hai', 0),

    # --- TOXIC: Explicit & Direct (Label 1) ---
    ('I hate you so much, you are terrible.', 1),
    ('Go away, nobody likes you.', 1),
    ('You are an idiot and a loser.', 1),
    ('Shut up, you are so annoying.', 1),
    ('You suck and your code is garbage.', 1),
    ('Get lost, you piece of trash.', 1),
    ('Die please, you are worthless.', 1),
    ('Kill yourself.', 1),
    ('I will hurt you.', 1),
    ('Stupid moron.', 1),
    ('Fuck off.', 1),
    ('You are a complete failure.', 1),
    ('Your existence is a mistake.', 1),
    ('Go rot in hell.', 1),
    ('You make me sick.', 1),

    # --- TOXIC: Obfuscated / Typos / Bypass attempts (Label 1) ---
    # (TF-IDF needs these exact character sequences to build its vocabulary)
    ('b!tch please', 1), 
    ('You are a total a$$hole.', 1), 
    ('sh1t', 1), 
    ('f u c k y o u', 1),
    ('f*** off', 1),
    ('stfu', 1),
    ('gtfo', 1),
    ('kys', 1), # Internet slang for 'kill yourself'
    ('ret*rd', 1),
    ('shiuty', 1),

    # --- TOXIC: Subtle / Gaslighting / Microaggressions (Label 1) ---
    ('You are overreacting, it is not a big deal.', 1),
    ('If you really cared, you would do this.', 1),
    ('I am just being honest, if you do not like it, that is your problem.', 1),
    ('You are so well-spoken for someone like you.', 1),
    ('Why are you being so sensitive?', 1),
    ('I was just joking, stop being a baby.', 1),
    ('You are lucky I even talk to you.', 1),
    ('Maybe try using your brain for once?', 1),
    ('Your ignorance is truly impressive.', 1),
    ('Bless your heart, you actually tried.', 1),
    ('With all due respect, you have no idea what you are talking about.', 1),

    # --- TOXIC: Hinglish / Multilingual Profanity (Label 1) ---
    ('chutiya', 1), 
    ('tu pagal hai kya', 1), 
    ('madarchod', 1), 
    ('bhosdike', 1),
    ('gandu', 1), 
    ('bhenchod', 1), 
    ('kaminey', 1), 
    ('saale', 1),
    ('bakwas mat kar', 1),
    ('tere baap ka naukar nahi hu', 1),
    ('aukat me reh', 1)
]

df = pd.DataFrame(raw_data, columns=['text', 'is_toxic'])

def train_model():
    print(f"Training refined model with {len(df)} examples...")
    X = df['text']
    y = df['is_toxic']

    # Vectorization with character n-grams for typo resilience
    tfidf = TfidfVectorizer(
        stop_words='english', 
        lowercase=True, 
        ngram_range=(1, 4), 
        analyzer='word',
        max_features=5000
    )
    X_tfidf = tfidf.fit_transform(X)

    model = LogisticRegression(
        class_weight='balanced', 
        C=0.8, # Slightly higher C for more confidence on this specific dataset
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
    
    print("Refined model and vectorizer saved to models/ folder.")

if __name__ == "__main__":
    train_model()
