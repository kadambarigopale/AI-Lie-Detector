import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.ml.nlp_utils import preprocess_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

def create_dummy_dataset() -> pd.DataFrame:
    """
    Creates a dummy dataset if no real dataset is provided.
    In a real-world scenario, you would load a CSV here.
    """
    data = [
        {"text": "The product is amazing! I use it every day and it works perfectly. Best purchase ever.", "label": "truthful"},
        {"text": "I highly recommend this item. The quality is exceptional and shipping was fast.", "label": "truthful"},
        {"text": "This is completely garbage. Broke within a day of use. Do not buy.", "label": "truthful"},
        {"text": "Worst experience ever. The customer service ignored me.", "label": "truthful"},
        
        {"text": "I absolutely love this product it changed my whole life completely and I am so happy", "label": "deceptive"},
        {"text": "Amazing incredible product, better than anything else in the world 10/10 perfect flawless.", "label": "deceptive"},
        {"text": "Do not buy this it is a scam total rip off worst thing ever made in history.", "label": "deceptive"},
        {"text": "I give this 5 stars because it is simply the best thing I have ever bought without a doubt.", "label": "deceptive"},
    ]
    # Replicate to get a reasonable training size
    return pd.DataFrame(data * 10)

def train_model(csv_path: str = None):
    print("Loading data...")
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("No CSV provided or found, using dummy dataset.")
        df = create_dummy_dataset()
        
    print("Preprocessing text...")
    
    # Handle different column names for labels
    if 'label' not in df.columns and 'deceptive' in df.columns:
        df.rename(columns={'deceptive': 'label'}, inplace=True)
        
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    X = df['clean_text']
    y = df['label']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression Model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_vec)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Save Model and Vectorizer
    print("Saving model to disk...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Training Complete!")

if __name__ == "__main__":
    # Point to the dataset at the project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Prefer the enriched dataset if it exists
    dataset_path = os.path.join(base_dir, 'dataset_with_emotions.csv')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, 'dataset.csv')
        
    train_model(dataset_path)
