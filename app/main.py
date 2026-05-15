import os
import pickle
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from app.ml.nlp_utils import preprocess_text, enrich_text_with_emotions, get_sentiment, get_emotion_scores, get_suspicious_words
from app.core.logger import logger

app = FastAPI(
    title="AI Lie Detection API",
    description="An API to classify text as Truthful or Deceptive",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'vectorizer.pkl')
BERT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'ml', 'bert_model')

# Global variables for caching model and vectorizer
model = None
vectorizer = None
bert_model = None
bert_tokenizer = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    sentiment: str = "Neutral"
    suspicious_words: list = []
    emotion_scores: dict = {}

@app.on_event("startup")
def load_artifacts():
    global model, vectorizer, bert_model, bert_tokenizer
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info("Logistic Regression model and vectorizer loaded successfully.")
        else:
            logger.warning("Logistic Regression model or vectorizer not found. Please run model_training.py first.")
            
        if os.path.exists(BERT_MODEL_DIR):
            logger.info("Loading BERT model and tokenizer...")
            bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
            bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
            bert_model.eval() # Set model to evaluation mode
            logger.info("BERT model and tokenizer loaded successfully.")
        else:
            logger.warning(f"BERT model not found at {BERT_MODEL_DIR}. Run bert_training.py to enable the BERT endpoint.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict_text(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    if not request.text.strip():
        logger.warning("Received empty text for prediction.")
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    logger.info(f"Received prediction request for text (length {len(request.text)}): {request.text[:50]}...")
    clean_text = preprocess_text(request.text)
    
    if not clean_text:
        # Edge case where preprocessing removes all words (e.g. text only contained numbers/stopwords)
        return PredictionResponse(prediction="unknown", confidence=0.0)
    
    vec_text = vectorizer.transform([clean_text])
    
    prediction = model.predict(vec_text)[0]
    
    probabilities = model.predict_proba(vec_text)[0]
    confidence = float(max(probabilities))
    
    sentiment = get_sentiment(request.text)
    suspicious_words = get_suspicious_words(request.text)
    emotion_scores = get_emotion_scores(request.text)
    
    logger.info(f"Prediction result: {prediction} (Confidence: {confidence:.2f})")
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        sentiment=sentiment,
        suspicious_words=suspicious_words,
        emotion_scores=emotion_scores
    )

@app.post("/predict/bert", response_model=PredictionResponse)
def predict_text_bert(request: PredictionRequest):
    if bert_model is None or bert_tokenizer is None:
        raise HTTPException(status_code=500, detail="BERT model is not loaded. Please train it first.")
    
    if not request.text.strip():
        logger.warning("Received empty text for BERT prediction.")
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    logger.info(f"Received BERT prediction request for text (length {len(request.text)}): {request.text[:50]}...")
    
    # Enrich text with sentiment and emotion
    enriched_text = enrich_text_with_emotions(request.text)
    logger.info(f"Enriched text for BERT: {enriched_text}")
    
    # Tokenize input
    inputs = bert_tokenizer(
        enriched_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # Predict
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        confidence, predicted_class_id = torch.max(probabilities, dim=-1)
        
    # Map index back to class name (0 -> truthful, 1 -> deceptive)
    prediction = "truthful" if predicted_class_id.item() == 0 else "deceptive"
    confidence_val = confidence.item()
    
    sentiment = get_sentiment(request.text)
    suspicious_words = get_suspicious_words(request.text)
    emotion_scores = get_emotion_scores(request.text)
    
    logger.info(f"BERT Prediction result: {prediction} (Confidence: {confidence_val:.2f})")
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence_val,
        sentiment=sentiment,
        suspicious_words=suspicious_words,
        emotion_scores=emotion_scores
    )

@app.get("/")
def read_root():
    return RedirectResponse(url="/index.html")

# Serve the frontend directory statically
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")
else:
    logger.error(f"Frontend directory not found at: {FRONTEND_DIR}")
