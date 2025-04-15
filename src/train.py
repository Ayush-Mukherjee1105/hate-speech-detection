import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    Trainer, TrainingArguments
)
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and preprocess the dataset
def load_and_preprocess_dataset(file_path):
    # Read the raw text file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse text and labels
    texts, labels = [], []
    label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}
    for line in lines:
        # Extract text and label from each line
        parts = line.strip().rsplit(",", 1)  # Split by last comma
        if len(parts) == 2:
            text, label = parts
            if label.lower() in label_map:
                texts.append(text.strip())
                labels.append(label_map[label.lower()])

    # Create a DataFrame
    df = pd.DataFrame({"text": texts, "label": labels})
    return df

# Load the dataset
file_path = "data\processed\hate_speech_augmented.csv"
df = load_and_preprocess_dataset(file_path)

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Function to train a model
def train_model(model_name):
    # Initialize tokenizer and model based on model_name
    if "mbert" in model_name.lower():
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3).to(device)
    elif "xlm-r" in model_name.lower():
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_data = {"text": train_texts, "label": train_labels}
    val_data = {"text": val_texts, "label": val_labels}

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Rename the 'label' column to 'labels'
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    # Set the format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training setup
    training_args = TrainingArguments(
        output_dir=f"models/fine_tuned/{model_name}",
        per_device_train_batch_size=32,  # Batch size for training
        per_device_eval_batch_size=32,  # Batch size for evaluation
        num_train_epochs=100,           # Number of epochs
        evaluation_strategy="epoch",   # Evaluate at the end of each epoch
        logging_dir=f"logs/{model_name}",
        logging_steps=50,              # Log every 50 steps
        save_strategy="epoch",         # Save checkpoint at the end of each epoch
        load_best_model_at_end=True,   # Load the best model at the end of training
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True if torch.cuda.is_available() else False  # Enable mixed precision on GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Save the final model
    model.save_pretrained(f"models/fine_tuned/{model_name}")
    print(f"Model saved to models/fine_tuned/{model_name}")

# Train both models
train_model("mbert")  # Train multilingual BERT
train_model("xlm-r")  # Train XLM-RoBERTa