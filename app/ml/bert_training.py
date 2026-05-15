import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Determine absolute path to the dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset_with_emotions.csv')
BERT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'bert_model')

def train_bert(csv_path: str = None, epochs: int = 3):
    print("Loading data...")
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"No dataset provided or found at {csv_path}. Cannot train BERT without data.")
        return

    print("Preprocessing data for BERT...")
    
    # Check for correct columns

    if 'text' not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
        
    # We train on the raw text directly so the model learns the linguistic cues.
    
    label_col = 'label'
    if 'label' not in df.columns:
        if 'deceptive' in df.columns:
            label_col = 'deceptive'
        else:
            # Let's see if the first column contains the labels based on dataset.csv
            label_col = df.columns[0]
            
    print(f"Using '{label_col}' as label column.")

    # Convert labels to integer: truthful=0, deceptive=1
    def map_label(val):
        if str(val).lower().strip() == 'truthful':
            return 0
        elif str(val).lower().strip() == 'deceptive':
            return 1
        return 0

    df['label'] = df[label_col].apply(map_label)
    df = df[['text', 'label']]
    
    # Shuffle and split data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    eval_df = df.iloc[split_idx:]

    print("Converting to Hugging Face Dataset...")
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    print("Loading Tokenizer...")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    print("Loading BERT Model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=BERT_MODEL_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8, # keep batch size small to prevent OOM
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting Training...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print(f"Evaluation Results: {results}")

    print("Saving trained model and tokenizer...")
    os.makedirs(BERT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(BERT_MODEL_DIR, safe_serialization=False)
    tokenizer.save_pretrained(BERT_MODEL_DIR)
    print(f"Model saved to {BERT_MODEL_DIR}")

if __name__ == "__main__":
    train_bert(DATASET_PATH, epochs=3)
