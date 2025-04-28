import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification
)
import torch
import numpy as np
from tqdm import tqdm

# Force CPU usage to avoid crashes
device = torch.device("cpu")
print(f"Using device: {device}")

# Load test data
df = pd.read_csv("data/processed/hate_speech_augmented.csv")

# Correct label map (same as your training)
label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}
df["label_mapped"] = df["label"].map(label_map)

# Drop rows where mapping failed
before = len(df)
df = df.dropna(subset=["label_mapped"])
after = len(df)
if before != after:
    print(f"Warning: Dropped {before - after} rows due to unmapped labels.")

test_texts = df["text"].tolist()
test_labels = df["label_mapped"].astype(int).tolist()

# Evaluate model safely
def evaluate_model(model_name, batch_size=16):  # <-- tiny batch
    if model_name == "mbert":
        tokenizer = BertTokenizer.from_pretrained("models/fine_tuned/mbert")
        model = BertForSequenceClassification.from_pretrained("models/fine_tuned/mbert").to(device)
    else:
        tokenizer = XLMRobertaTokenizer.from_pretrained("models/fine_tuned/xlm-r")
        model = XLMRobertaForSequenceClassification.from_pretrained("models/fine_tuned/xlm-r").to(device)

    model.eval()
    preds = []

    for i in tqdm(range(0, len(test_texts), batch_size), desc=f"Evaluating {model_name.upper()}"):
        batch_texts = test_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.softmax(outputs.logits, dim=-1).argmax(dim=1).cpu().tolist()
            preds.extend(batch_preds)

    return classification_report(test_labels, preds, output_dict=True)

# Evaluate both models
mbert_report = evaluate_model("mbert")
xlmr_report = evaluate_model("xlm-r")

# Plot comparison
metrics = ["precision", "recall", "f1-score"]
models = ["mBERT", "XLM-R"]

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.35
index = np.arange(len(metrics))

colors = ["#4C72B0", "#55A868"]  # Calmer blue-green tones
for i, model in enumerate(models):
    scores = [mbert_report["weighted avg"][m] if model == "mBERT" else xlmr_report["weighted avg"][m] for m in metrics]
    ax.bar(index + i * bar_width, scores, bar_width, label=model, color=colors[i])

ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Model Comparison: mBERT vs XLM-R', fontsize=16)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(["Precision", "Recall", "F1-score"], fontsize=12)
ax.set_ylim(0, 1)
ax.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
