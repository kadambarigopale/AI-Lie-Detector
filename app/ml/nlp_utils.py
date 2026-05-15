import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# We might need to download nltk data if not already present.
# In a real setup, you would ensure this runs once or is part of a build script.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Cleans the input text for ML processing:
    - Lowercase
    - Remove punctuation and special characters
    - Remove stopwords
    - Lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize (simple whitespace split)
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]
    
    return " ".join(clean_tokens)

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment(text: str) -> str:
    scores = sia.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.5:
        return 'Highly Positive'
    elif compound >= 0.05:
        return 'Positive'
    elif compound <= -0.5:
        return 'Strongly Negative'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_emotional_tone(text: str) -> str:
    scores = get_emotion_scores(text)
    max_emotion = max(scores, key=scores.get)
    if scores[max_emotion] == 0:
        return 'neutral'
    return max_emotion

def get_emotion_scores(text: str) -> dict:
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
                
    return scores

def get_suspicious_words(text: str) -> list:
    """Heuristic logic to find words often correlated with deception/over-justification."""
    text_lower = str(text).lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    suspicious_lexicon = {
        'honestly', 'truthfully', 'frankly', 'literally', 'swear', 'promise', 
        'never', 'always', 'absolutely', 'certainly', 'basically', 'actually',
        'just', 'uh', 'um', 'ah', 'like', 'stuff', 'things', 'seriously', 
        'really', 'truly', 'trust', 'believe', 'obviously', 'clearly', 
        'simply', 'maybe', 'perhaps', 'possibly', 'somewhat', 'kinda', 
        'sorta', 'mostly', 'exactly', 'purely', 'evidently', 'lied', 'lying'
    }
    
    found = []
    for w in words:
        if w in suspicious_lexicon and w not in found:
            found.append(w)
            
    return found

def enrich_text_with_emotions(text: str) -> str:
    """Prepends sentiment and emotional tone to the text for BERT input."""
    sentiment = get_sentiment(text)
    emotion = get_emotional_tone(text)
    return f"Sentiment: {sentiment}. Emotion: {emotion}. {text}"
