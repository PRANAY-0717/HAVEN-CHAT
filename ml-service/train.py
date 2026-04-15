import os
import pickle
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model():
    print("Fetching dataset from Hugging Face...")
    # Load the "hate_speech_offensive" dataset (contains ~24k real tweets)
    dataset = load_dataset("hate_speech_offensive", split="train")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)
    
    # Map the labels to binary (0 = Non-Toxic, 1 = Toxic)
    # The original dataset classes: 0 = Hate speech, 1 = Offensive, 2 = Neither
    df['is_toxic'] = df['class'].apply(lambda x: 0 if x == 2 else 1)
    df = df.rename(columns={'tweet': 'text'})
    
    # Randomly sample exactly 10,000 rows to keep training fast but accurate
    print("Sampling 10,000 rows...")
    df = df[['text', 'is_toxic']].sample(10000, random_state=42)
    
    X = df['text']
    y = df['is_toxic']
    
    print(f"Training model with {len(df)} examples...")

    # Vectorization optimized for real-world messy text
    tfidf = TfidfVectorizer(
        lowercase=True, 
        analyzer='word',
        ngram_range=(1, 2), # Captures single words and pairs (e.g., "not bad")
        max_features=10000, # Increased features since we have 10k rows now
        # Custom token pattern to keep important symbols (like $ or *) used in obfuscated slurs
        token_pattern=r'(?u)\b\w+\b|[!@#$%^&*]+'
    )
    
    X_tfidf = tfidf.fit_transform(X)

    # Train Logistic Regression
    model = LogisticRegression(
        class_weight='balanced', 
        C=1.0, # Standard regularization
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_tfidf, y)

    # Save the artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    print("✅ Model trained successfully on 10,000 real-world examples!")
    print("Artifacts saved to models/model.pkl and models/vectorizer.pkl")

if __name__ == "__main__":
    train_model()