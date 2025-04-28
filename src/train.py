# src/train.py
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("data/processed/hasoc2021_multilingual.csv")

# Train-test split (stratify by label)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Tokenize function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Trainer wrapper
def train_model(model_name, tokenizer_class, model_class, pretrained_name):
    print(f"\n=== Training {model_name} ===")
    tokenizer = tokenizer_class.from_pretrained(pretrained_name)
    model = model_class.from_pretrained(pretrained_name, num_labels=2).to(device)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=f"models/fine_tuned/{model_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        learning_rate=2e-5,                     # ⬅️ tuned
        weight_decay=0.01,                      # ⬅️ regularization
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,                     # ⬅️ increased slightly
        warmup_steps=500,                       # ⬅️ to slow down LR decay
        logging_dir=f"logs/{model_name}",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,           # ⬅️ effective larger batch
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(f"models/fine_tuned/{model_name}")
    tokenizer.save_pretrained(f"models/fine_tuned/{model_name}")
    print(f"Saved model to models/fine_tuned/{model_name}")

# Train both models
train_model(
    model_name="mbert",
    tokenizer_class=BertTokenizer,
    model_class=BertForSequenceClassification,
    pretrained_name="bert-base-multilingual-cased"
)

train_model(
    model_name="xlmr",
    tokenizer_class=XLMRobertaTokenizer,
    model_class=XLMRobertaForSequenceClassification,
    pretrained_name="xlm-roberta-base"
)
