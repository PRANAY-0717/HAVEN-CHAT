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
        'I love this project, it is amazing!',
        'You are a great person.',
        'Have a nice day!',
        'This is a helpful tool.',
        'I hate you so much, you are terrible.',
        'Go away, nobody likes you.',
        'You are an idiot and a loser.',
        'Shut up, you are so annoying.',
        'This is the best thing ever!',
        'You suck and your code is garbage.',
        'Hello there, how are you?',
        'I disagree with your opinion but respect it.',
        'This is unacceptable behavior.',
        'You are doing a fantastic job!',
        'Get lost, you piece of trash.',
        'Die please, you are worthless.',
        'Kill yourself.',
        'I will hurt you.',
        'You should not be alive.',
        'Stupid moron.',
        'Fuck off.',
        'You are a complete failure.',
        'You are the best.',
        'You are amazing.',
        'I really like you.',
        'This is wonderful.',
        'You are so kind.',
        'Thank you for your help.',
        'Have a beautiful day.',
        'I appreciate your hard work.',
        'You are doing great.',
        'I love you.',
        'This is great news.',
        'You are a hero.',
        'Keep up the good work.'
    ],
    'is_toxic': [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

def train_model():
    print("Training model...")
    X = df['text']
    y = df['is_toxic']

    # Vectorization
    tfidf = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X)

    # Model with balanced weights and better regularization
    model = LogisticRegression(class_weight='balanced', C=1.0)
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
