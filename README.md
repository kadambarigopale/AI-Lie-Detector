# 🕵️‍♂️ AI Lie Detector: Advanced Deception Analysis System

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co/docs/transformers/index)
[![BERT](https://img.shields.io/badge/BERT-4285F4?style=for-the-badge&logo=google)](https://github.com/google-research/bert)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An end-to-end, high-performance AI system designed to detect deception in text using state-of-the-art Natural Language Processing (NLP). This project combines traditional Machine Learning with Deep Learning (BERT) to provide highly accurate classification and granular emotional insights.

## ✨ Key Features

- **Hybrid Prediction Pipeline**: 
  - **Logistic Regression + TF-IDF**: Fast, interpretable baseline for text classification.
  - **BERT (Bidirectional Encoder Representations from Transformers)**: Fine-tuned for nuanced deception detection using context-aware embeddings.
- **Multidimensional Analysis**:
  - **Emotion Detection**: Extracts Joy, Anger, Fear, Sadness, etc., to identify emotional inconsistencies.
  - **Sentiment Scoring**: Real-time sentiment intensity analysis using VADER.
  - **Suspicious Word Highlighting**: Identifies linguistic markers frequently associated with deceptive behavior.
- **Interactive Visual Dashboard**: A premium, glassmorphic UI featuring:
  - Confidence gauges and probability distributions.
  - Real-time emotional tone mapping.
  - Highlighted suspicious keywords within the input text.
- **Enriched Input Processing**: The BERT model leverages "Emotional Enrichment," where sentiment and emotion tags are prepended to the input text to improve classification accuracy.

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Machine Learning**: Scikit-learn, PyTorch, Hugging Face Transformers
- **NLP**: NLTK, BERT, VADER
- **Frontend**: Vanilla JavaScript, HTML5, CSS3 (Glassmorphism design)
- **DevOps**: UV (Package management), Pytest (Testing)

## 📁 Project Structure

```text
ai-lie-detector/
├── app/
│   ├── core/           # Centralized logging & configuration
│   ├── ml/             # ML Models, Tokenizers, and Training scripts
│   │   ├── bert_training.py   # BERT fine-tuning pipeline
│   │   ├── model_training.py  # Logistic Regression training
│   │   └── nlp_utils.py       # Preprocessing & Emotion analysis
│   └── main.py         # FastAPI Entrypoint & Routes
├── frontend/           # Web-based Interactive Dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/              # Automated Test Suite
├── dataset.csv         # Core training data
└── requirements.txt    # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- (Optional) CUDA-enabled GPU for faster BERT training/inference

### 1. Installation
Clone the repository and install dependencies using `pip` or `uv`:
```bash
# Using pip
pip install -r requirements.txt

# Using uv (Recommended)
uv sync
```

### 2. Model Training
Before running the app, ensure the models are trained:
```bash
# Train Logistic Regression
python -m app.ml.model_training

# Train BERT (requires significant compute/time)
python -m app.ml.bert_training
```

### 3. Running the Application
The system is integrated into a single FastAPI application that serves both the API and the static frontend.
```bash
uvicorn app.main:app --reload
```
- **Backend API**: `http://localhost:8000/docs`
- **Frontend Dashboard**: `http://localhost:8000/`

## 🧪 Testing
Run the automated test suite to verify system integrity:
```bash
python -m pytest tests/
```

## 📊 How it Works
The system uses a unique **Contextual Enrichment** approach. Instead of analyzing raw text, the input is first passed through an NLP pipeline that extracts emotional features. These features are then concatenated with the original text before being processed by the BERT classifier, allowing the model to "understand" the emotional context of the deception.

---
Developed with ❤️ for Advanced Deception Research.
