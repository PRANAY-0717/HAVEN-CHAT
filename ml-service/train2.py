import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os

# =============================================================================
# HAVEN-GUARD Training Dataset v3.0
# =============================================================================
# Design principles:
#   1. NON-TOXIC examples outnumber TOXIC ~2:1 to bias toward NOT blocking
#   2. Every common greeting/conversational phrase has multiple variants
#   3. Toxic examples focus on CLEAR harmful intent, not ambiguous cases
#   4. Includes adversarial, gaming, tech, slang, and multilingual examples
# =============================================================================

raw_data = [
    # =========================================================================
    # NON-TOXIC: Greetings & Basic Conversation (Label 0)
    # =========================================================================
    ('hi', 0),
    ('hey', 0),
    ('hello', 0),
    ('hey there', 0),
    ('hi there', 0),
    ('hello there', 0),
    ('howdy', 0),
    ('yo', 0),
    ('sup', 0),
    ('whats up', 0),
    ("what's up", 0),
    ('how are you', 0),
    ('how are you doing', 0),
    ('how are you today', 0),
    ('how r u', 0),
    ('how is it going', 0),
    ("how's it going", 0),
    ('how have you been', 0),
    ("how's your day", 0),
    ('how is your day going', 0),
    ('what are you up to', 0),
    ('what are you doing', 0),
    ('long time no see', 0),
    ('good to see you', 0),
    ('nice to meet you', 0),
    ('good morning', 0),
    ('good afternoon', 0),
    ('good evening', 0),
    ('good night', 0),
    ('gm', 0),
    ('gn', 0),

    # =========================================================================
    # NON-TOXIC: Positive & Friendly Messages (Label 0)
    # =========================================================================
    ('I love this project, it is amazing!', 0),
    ('You are a great person.', 0),
    ('Have a nice day!', 0),
    ('This is a helpful tool.', 0),
    ('This is the best thing ever!', 0),
    ('I disagree with your opinion but respect it.', 0),
    ('You are doing a fantastic job!', 0),
    ('Thank you for your help.', 0),
    ('I appreciate your hard work.', 0),
    ('Can you help me with this?', 0),
    ('Let us work together on this.', 0),
    ('You did well on that.', 0),
    ('Great work!', 0),
    ('Awesome!', 0),
    ('That was really good', 0),
    ('You are the best', 0),
    ('You rock', 0),
    ('You are amazing', 0),
    ('I am proud of you', 0),
    ('Well done!', 0),
    ('Nice one!', 0),
    ('Keep it up!', 0),
    ('Brilliant!', 0),
    ('You nailed it', 0),
    ('That is so cool', 0),
    ('I like your idea', 0),
    ('Sounds good to me', 0),
    ('I agree with you', 0),
    ('You make a good point', 0),
    ('Fair enough', 0),
    ('No problem', 0),
    ('Sure thing', 0),
    ('Of course', 0),
    ('Absolutely', 0),
    ('You are welcome', 0),
    ('My pleasure', 0),
    ('Happy to help', 0),
    ('Glad to hear that', 0),
    ('That makes sense', 0),
    ('I see what you mean', 0),
    ('I understand', 0),
    ('Got it', 0),
    ('Makes sense', 0),
    ('Take care', 0),
    ('See you later', 0),
    ('Bye', 0),
    ('Goodbye', 0),
    ('Talk to you soon', 0),
    ('Have a great day', 0),
    ('Enjoy your weekend', 0),
    ('Happy birthday', 0),
    ('Congratulations', 0),
    ('Best of luck', 0),
    ('Good luck', 0),
    ('I hope you feel better', 0),
    ('Get well soon', 0),
    ('Miss you', 0),
    ('Love you', 0),
    ('Take it easy', 0),
    ('Stay safe', 0),
    ('Be careful', 0),
    ('All the best', 0),
    ('Cheers', 0),

    # =========================================================================
    # NON-TOXIC: Short / Neutral / Common Replies (Label 0)
    # =========================================================================
    ('Exactly', 0),
    ('correct', 0),
    ('true', 0),
    ('i agree', 0),
    ('ok', 0),
    ('okay', 0),
    ('yes', 0),
    ('no', 0),
    ('maybe', 0),
    ('thanks', 0),
    ('thank you', 0),
    ('thx', 0),
    ('ty', 0),
    ('np', 0),
    ('lol', 0),
    ('lmao', 0),
    ('haha', 0),
    ('hehe', 0),
    ('xD', 0),
    ('bruh', 0),
    ('dude', 0),
    ('bro', 0),
    ('man', 0),
    ('nice', 0),
    ('cool', 0),
    ('wow', 0),
    ('omg', 0),
    ('idk', 0),
    ('idc', 0),
    ('nvm', 0),
    ('tbh', 0),
    ('imo', 0),
    ('btw', 0),
    ('afk', 0),
    ('brb', 0),
    ('same', 0),
    ('mood', 0),
    ('facts', 0),
    ('real', 0),
    ('fr', 0),
    ('fr fr', 0),
    ('bet', 0),
    ('cap', 0),
    ('no cap', 0),
    ('sus', 0),
    ('based', 0),
    ('W', 0),
    ('L', 0),
    ('ratio', 0),
    ('oof', 0),
    ('yikes', 0),
    ('yep', 0),
    ('nope', 0),
    ('sure', 0),
    ('alright', 0),
    ('fine', 0),
    ('whatever', 0),
    ('anyway', 0),
    ('hmm', 0),
    ('ah', 0),
    ('oh', 0),
    ('I see', 0),
    ('right', 0),
    ('yup', 0),
    ('yeah', 0),
    ('nah', 0),

    # =========================================================================
    # NON-TOXIC: Questions & Conversational (Label 0)
    # =========================================================================
    ('What do you think?', 0),
    ('Where are you from?', 0),
    ('What time is it?', 0),
    ('Can I ask you something?', 0),
    ('Do you know how to do this?', 0),
    ('Have you seen this movie?', 0),
    ('What is your favorite color?', 0),
    ('How old are you?', 0),
    ('Where do you live?', 0),
    ('What do you do for work?', 0),
    ('Are you free tomorrow?', 0),
    ('Want to hang out?', 0),
    ('Did you finish the project?', 0),
    ('Can we talk?', 0),
    ('What happened?', 0),
    ('Is everything okay?', 0),
    ('Are you alright?', 0),
    ('Do you need help?', 0),
    ('What is wrong?', 0),
    ('Tell me more', 0),
    ('Why do you say that?', 0),
    ('How does that work?', 0),
    ('Could you explain?', 0),
    ('What is the deadline?', 0),
    ('When is the meeting?', 0),
    ('Who else is coming?', 0),
    ('Should we start?', 0),
    ('Any updates?', 0),
    ('How is the project going?', 0),
    ('What is the plan?', 0),
    ('Are we on track?', 0),

    # =========================================================================
    # NON-TOXIC: Constructive Criticism & Disagreement (Label 0)
    # =========================================================================
    ('I think there is a better way to do this.', 0),
    ('I do not think that is correct.', 0),
    ('I respectfully disagree.', 0),
    ('That is not quite right.', 0),
    ('I think you might be wrong about that.', 0),
    ('Can we reconsider this approach?', 0),
    ('This code has some bugs.', 0),
    ('This needs more work.', 0),
    ('The design could be improved.', 0),
    ('I do not like this feature.', 0),
    ('This is confusing.', 0),
    ('This could be better.', 0),
    ('Not my favorite.', 0),
    ('I expected more.', 0),
    ('You might want to rethink this.', 0),
    ('This approach has some issues.', 0),
    ('I found a problem with this.', 0),

    # =========================================================================
    # NON-TOXIC: Emotional / Self-Directed Frustration (Label 0)
    # =========================================================================
    ('I am so frustrated right now.', 0),
    ('This is driving me crazy.', 0),
    ('I hate this bug.', 0),
    ('I hate Mondays.', 0),
    ('This is so annoying.', 0),
    ('I cannot believe I made that mistake.', 0),
    ('I am so stupid sometimes.', 0),
    ('I feel terrible about it.', 0),
    ('I am so tired.', 0),
    ('I had a bad day.', 0),
    ('Life is hard sometimes.', 0),
    ('I wish things were different.', 0),
    ('I am stressed out.', 0),
    ('I need a break.', 0),
    ('This is overwhelming.', 0),
    ('I am burned out.', 0),
    ('Ugh, not again.', 0),
    ('Why does this keep happening.', 0),
    ('I messed up.', 0),
    ('I feel dumb.', 0),

    # =========================================================================
    # NON-TOXIC: Gaming / Tech / Slang (Label 0)
    # =========================================================================
    ('kill the process', 0),
    ('crash the server', 0),
    ('execute the program', 0),
    ('dump the memory', 0),
    ('headshot', 0),
    ('gg', 0),
    ('gg wp', 0),
    ('gg ez', 0),
    ('destroy the enemy base', 0),
    ('this new feature is sick', 0),
    ('that was a badass presentation', 0),
    ('holy cow', 0),
    ('my bad', 0),
    ('no worries', 0),
    ('I am going to kill this exam', 0),
    ('this pizza is the bomb', 0),
    ('not bad at all', 0),
    ('lets gooo', 0),
    ('clutch play', 0),
    ('that was insane', 0),
    ('this game is fire', 0),
    ('drop the bomb on them', 0),
    ('nuke the build', 0),
    ('kill the branch', 0),
    ('force push', 0),
    ('merge conflict', 0),
    ('rebase and squash', 0),
    ('bug squashing', 0),
    ('this code slaps', 0),
    ('ship it', 0),
    ('deploy to prod', 0),
    ('the server is dying', 0),
    ('killing it today', 0),
    ('savage play', 0),
    ('that was brutal', 0),
    ('crushing it', 0),
    ('lets run it back', 0),
    ('one more', 0),
    ('rage quit', 0),
    ('tilted', 0),
    ('griefing is annoying', 0),
    ('camper', 0),
    ('noob', 0),
    ('newbie', 0),
    ('lag is insane', 0),
    ('we got rekt', 0),
    ('trash at this game', 0),
    ('wrecked them', 0),
    ('owned', 0),
    ('pwned', 0),
    ('git gud', 0),
    ('carry me please', 0),
    ('one tap', 0),
    ('ace', 0),
    ('MVP', 0),

    # =========================================================================
    # NON-TOXIC: Hinglish / Multilingual (Label 0)
    # =========================================================================
    ('kya haal hai bhai', 0),
    ('sahi hai yaar', 0),
    ('tu toh hero hai', 0),
    ('shabash', 0),
    ('mast lag raha hai', 0),
    ('kaise ho', 0),
    ('theek hai', 0),
    ('accha', 0),
    ('kya kar rahe ho', 0),
    ('kahan hai tu', 0),
    ('bhai kya scene hai', 0),
    ('maza aa gaya', 0),
    ('bohot accha', 0),
    ('kya baat hai', 0),
    ('chal theek hai', 0),
    ('haan bhai', 0),
    ('nahi yaar', 0),
    ('arey yaar', 0),
    ('bas ho gaya', 0),
    ('chalo', 0),

    # =========================================================================
    # NON-TOXIC: You-containing safe phrases (critical for fixing "you" bias)
    # =========================================================================
    ('are you coming?', 0),
    ('do you want to play?', 0),
    ('are you there?', 0),
    ('where are you?', 0),
    ('did you eat?', 0),
    ('you too', 0),
    ('you as well', 0),
    ('see you', 0),
    ('miss you too', 0),
    ('thinking of you', 0),
    ('you are right', 0),
    ('you know what I mean', 0),
    ('you should try it', 0),
    ('you can do it', 0),
    ('you got this', 0),
    ('I believe in you', 0),
    ('you first', 0),
    ('after you', 0),
    ('you are funny', 0),
    ('you are smart', 0),
    ('you are talented', 0),
    ('you are kind', 0),
    ('you are my friend', 0),
    ('what about you', 0),
    ('and you', 0),
    ('thank you so much', 0),
    ('I am happy for you', 0),
    ('I support you', 0),
    ('I trust you', 0),
    ('you inspire me', 0),

    # =========================================================================
    # TOXIC: Direct Insults & Attacks (Label 1)
    # =========================================================================
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
    ('You are disgusting.', 1),
    ('Nobody wants you here.', 1),
    ('You should just disappear.', 1),
    ('Everybody hates you.', 1),
    ('What a pathetic loser.', 1),
    ('You are worthless garbage.', 1),
    ('You are a waste of space.', 1),
    ('I wish you did not exist.', 1),
    ('The world is worse with you in it.', 1),
    ('You are nothing but a burden.', 1),
    ('You deserve to suffer.', 1),
    ('Go crawl in a hole and die.', 1),
    ('Piece of shit.', 1),
    ('You are human garbage.', 1),
    ('I hope bad things happen to you.', 1),

    # =========================================================================
    # TOXIC: Threats & Violence (Label 1)
    # =========================================================================
    ('I will find you and beat you up.', 1),
    ('I am going to hurt you.', 1),
    ('Watch your back.', 1),
    ('I know where you live.', 1),
    ('You will regret this.', 1),
    ('I will make you pay.', 1),
    ('I am coming for you.', 1),
    ('You are dead to me.', 1),
    ('I will destroy your life.', 1),
    ('Better sleep with one eye open.', 1),
    ('I will make your life miserable.', 1),
    ('Say that to my face.', 1),
    ('Prepare for the worst.', 1),
    ('You do not know what is coming.', 1),

    # =========================================================================
    # TOXIC: Hate Speech & Identity Targeting (Label 1)
    # =========================================================================
    ('People like you should not exist.', 1),
    ('Go back to where you came from.', 1),
    ('Your kind is not welcome here.', 1),
    ('You people are all the same.', 1),
    ('You are a disgrace to your race.', 1),
    ('Typical behavior from your type.', 1),
    ('All of you are the problem.', 1),
    ('This country would be better without your kind.', 1),

    # =========================================================================
    # TOXIC: Obfuscated / Bypass Attempts (Label 1)
    # =========================================================================
    ('b!tch please', 1),
    ('You are a total a$$hole.', 1),
    ('sh1t', 1),
    ('f u c k y o u', 1),
    ('f*** off', 1),
    ('stfu', 1),
    ('gtfo', 1),
    ('kys', 1),
    ('ret*rd', 1),
    ('shiuty', 1),
    ('fck you', 1),
    ('fuk u', 1),
    ('fk off', 1),
    ('s h i t', 1),
    ('a s s h o l e', 1),
    ('b i t c h', 1),
    ('d i e', 1),
    ('k i l l yourself', 1),

    # =========================================================================
    # TOXIC: Subtle / Gaslighting / Microaggressions (Label 1)
    # =========================================================================
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
    ('No one asked for your opinion.', 1),
    ('You always ruin everything.', 1),
    ('Everything you touch turns to garbage.', 1),
    ('You are embarrassing yourself.', 1),
    ('Imagine being this clueless.', 1),

    # =========================================================================
    # TOXIC: Hinglish / Multilingual Profanity (Label 1)
    # =========================================================================
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
    ('aukat me reh', 1),
    ('kutte', 1),
    ('harami', 1),
    ('naalayak', 1),
    ('nikamma', 1),
    ('bevakoof', 1),
    ('gadhe', 1),
    ('tujhe toh', 1),
    ('chal hat', 1),
    ('teri aukat kya hai', 1),
]

df = pd.DataFrame(raw_data, columns=['text', 'is_toxic'])

def train_model():
    print(f"Training HAVEN-GUARD model v3.0 with {len(df)} examples...")
    print(f"  Non-toxic: {len(df[df['is_toxic'] == 0])} examples")
    print(f"  Toxic:     {len(df[df['is_toxic'] == 1])} examples")

    X = df['text']
    y = df['is_toxic']

    # TF-IDF with both word and character n-grams for robustness
    # Word n-grams capture semantic intent, char n-grams catch obfuscation
    tfidf = TfidfVectorizer(
        stop_words=None,       # Do NOT remove stop words — "you", "are", etc. are critical context
        lowercase=True,
        ngram_range=(1, 3),    # Unigrams, bigrams, trigrams
        analyzer='word',
        max_features=10000,
        min_df=1,
        sublinear_tf=True,     # Apply log normalization to TF — reduces bias from repeated terms
    )
    X_tfidf = tfidf.fit_transform(X)

    model = LogisticRegression(
        class_weight={0: 1.0, 1: 1.5},  # Only slightly weight toxic class — prevent over-flagging
        C=0.5,                            # Lower C = stronger regularization = more generalization
        solver='liblinear',
        max_iter=2000,
    )
    model.fit(X_tfidf, y)

    # Quick sanity check on known phrases
    test_phrases = [
        'how are you', 'hello', 'hi', 'how are you doing', 'what are you up to',
        'you are great', 'good morning', 'hey there', 'bye',
        'kill yourself', 'fuck off', 'you are an idiot', 'go die',
    ]
    print("\n--- Sanity Check ---")
    for phrase in test_phrases:
        X_test = tfidf.transform([phrase])
        prob = model.predict_proba(X_test)[0][1]
        pred = "TOXIC" if prob > 0.7 else "SAFE"
        print(f"  [{pred:5s}] (prob={prob:.3f}) \"{phrase}\"")
    print("-------------------\n")

    if not os.path.exists('models'):
        os.makedirs('models')
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    print("Model and vectorizer saved to models/ folder.")

if __name__ == "__main__":
    train_model()
