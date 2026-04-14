import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import os

# Sample dataset (In a real scenario, use Kaggle Jigsaw dataset)
data = {
    'text': [
        # Non-Toxic
        'I love this project, it is amazing!', 'You are a great person.', 'Have a nice day!', 
        'This is a helpful tool.', 'This is the best thing ever!', 'Hello there, how are you?', 
        'I disagree with your opinion but respect it.', 'You are doing a fantastic job!', 
        'You are the best.', 'You are amazing.', 'I really like you.', 'This is wonderful.', 
        'You are so kind.', 'Thank you for your help.', 'Have a beautiful day.', 
        'I appreciate your hard work.', 'You are doing great.', 'I love you.', 
        'This is great news.', 'You are a hero.', 'Keep up the good work.',
        'Can you help me with this?', 'What a wonderful afternoon.', 'The weather is lovely.',
        'I am so happy for you.', 'Congratulations on your success!', 'That is a very interesting point.',
        'I would like to learn more about this.', 'Let us work together on this.',
        
        # Toxic / Offensive
        'I hate you so much, you are terrible.', 'Go away, nobody likes you.', 
        'You are an idiot and a loser.', 'Shut up, you are so annoying.', 
        'You suck and your code is garbage.', 'Get lost, you piece of trash.', 
        'Die please, you are worthless.', 'Kill yourself.', 'I will hurt you.', 
        'You should not be alive.', 'Stupid moron.', 'Fuck off.', 'You are a complete failure.',
        'You are a disgusting pig.', 'I hope you fail in life.', 'You are a pathetic excuse for a human.',
        'Nobody wants you here.', 'Your existence is a mistake.', 'You are so incredibly stupid.',
        'Go rot in hell.', 'You are a useless waste of space.', 'I hate everything about you.',
        'You make me sick.', 'You are the worst person I have ever met.',
        
        # Edge cases / Sarcastic (often hard for models)
        'Oh great, another brilliant idea from you.', 'You must be so proud of your failure.',
        'I love how you think you know everything.', 'Wow, you are so smart (not).',
        'Maybe try using your brain for once?', 'Your ignorance is truly impressive.',
        
        # Slurs and Strong Language
        'Bitch please.', 'You are a total asshole.', 'What a dumbass.', 'Stop being such a prick.',
        'You are a coward and a liar.', 'Don\'t be such a jerk.', 'You are acting like a fool.'
    ],
    'is_toxic': [
        # Non-Toxic (0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Toxic (1)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Edge cases (1)
        1, 1, 1, 1, 1, 1,
        # Slurs (1)
        1, 1, 1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

def train_model():
    print("Training model...")
    X = df['text']
    y = df['is_toxic']

    # Vectorization: Character n-grams for typo resilience + word n-grams
    tfidf = TfidfVectorizer(
        stop_words='english', 
        lowercase=True, 
        ngram_range=(1, 3), 
        analyzer='word',
        max_features=5000
    )
    
    # We could also use a separate char-level vectorizer and combine them
    # but for simplicity, we'll stick to a robust word-level one with higher n-grams
    X_tfidf = tfidf.fit_transform(X)

    # Model with higher regularization strength (lower C) to prevent overfitting on small data
    # and class_weight='balanced' to handle any slight imbalances
    model = LogisticRegression(
        class_weight='balanced', 
        C=0.5, 
        solver='liblinear',
        max_iter=1000
    )
    model.fit(X_tfidf, y)

    # Save model and vectorizer
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    print("Model and vectorizer saved to models/ folder.")

if __name__ == "__main__":
    train_model()
