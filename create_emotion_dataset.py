import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def get_emotional_tone(text):
    text = str(text).lower()
    
    emotions = {
        'joy': ['happy', 'great', 'excellent', 'good', 'wonderful', 'fantastic', 'love', 'perfect', 'beautiful', 'glad', 'enjoy', 'pleased'],
        'anger': ['angry', 'terrible', 'awful', 'bad', 'horrible', 'worst', 'hate', 'annoyed', 'mad', 'rude', 'unacceptable'],
        'sadness': ['sad', 'disappointed', 'upset', 'depressed', 'sorry', 'cry', 'miserable', 'unhappy', 'regret'],
        'fear': ['scared', 'afraid', 'fear', 'terrified', 'creepy', 'worry', 'nervous', 'panic', 'threat'],
        'surprise': ['surprise', 'shock', 'amaze', 'astonish', 'unexpected', 'wow', 'sudden', 'unbelievable']
    }
    
    scores = {emotion: 0 for emotion in emotions}
    
    for emotion, keywords in emotions.items():
        for word in keywords:
            if word in text:
                scores[emotion] += 1
                
    # Find the emotion with the highest score
    max_emotion = max(scores, key=scores.get)
    
    # If no emotion keywords found, default to neutral
    if scores[max_emotion] == 0:
        return 'neutral'
    
    return max_emotion

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'dataset.csv')
    output_file = os.path.join(base_dir, 'dataset_with_emotions.csv')
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return
        
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    if 'text' not in df.columns:
        print("Error: 'text' column not found in dataset.")
        return
        
    print("Initializing Sentiment Analyzer...")
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        scores = sia.polarity_scores(str(text))
        compound = scores['compound']
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
            
    print("Calculating Sentiments...")
    df['sentiment'] = df['text'].apply(get_sentiment)
    
    print("Calculating Emotional Tones...")
    df['emotional_tone'] = df['text'].apply(get_emotional_tone)
    
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done! First 5 rows of new dataset:")
    print(df[['text', 'sentiment', 'emotional_tone']].head())

if __name__ == "__main__":
    main()
