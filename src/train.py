from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    Trainer, TrainingArguments
)
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and preprocess the dataset
def load_and_preprocess_dataset(file_path):
    """Load and preprocess the dataset from a CSV file."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    texts, labels = [], []
    label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}
    for line in lines:
        parts = line.strip().rsplit(",", 1)  # Split by the last comma
        if len(parts) == 2:
            text, label = parts
            if label.lower() in label_map:
                texts.append(text.strip())
                labels.append(label_map[label.lower()])
    return pd.DataFrame({"text": texts, "label": labels})

# Load dataset
file_path = "data/processed/hate_speech_augmented.csv"
df = load_and_preprocess_dataset(file_path)

def balance_dataset(df, label_col="label"):
    """Balance the dataset to have equal number of samples per class."""
    min_count = df[label_col].value_counts().min()
    print(f"Balancing dataset: each class will have {min_count} samples.")
    
    balanced_df = df.groupby(label_col).sample(n=min_count, random_state=42)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df

# Split into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Step 2: Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Step 3: Train the model
def train_model(model_name):
    """Train a model with the specified name."""

    # Initialize tokenizer and model
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

    train_data = {"text": train_texts.tolist(), "label": train_labels.tolist()}
    val_data = {"text": val_texts.tolist(), "label": val_labels.tolist()}

    train_dataset = Dataset.from_dict(train_data).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_dict(val_data).map(tokenize_function, batched=True)

    # Rename columns for Trainer
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    # Set format
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"models/fine_tuned/{model_name}",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        logging_dir=f"logs/{model_name}",
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(f"models/fine_tuned/{model_name}")
    tokenizer.save_pretrained(f"models/fine_tuned/{model_name}")
    print(f"Model and tokenizer saved to models/fine_tuned/{model_name}")

# Step 4: Train
train_model("mbert")   # Train multilingual BERT
train_model("xlm-r")   # Train XLM-RoBERTa
