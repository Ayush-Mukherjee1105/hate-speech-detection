import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd
from transformers import (
    BertTokenizer, BertForSequenceClassification,  # For mBERT
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification  # For XLM-R
)
import torch
import numpy as np

# Load test data
df = pd.read_csv("data/processed/hate_speech_augmented.csv")
test_texts = df["text"].tolist()
test_labels = df["label"].map({"hateful": 0, "non-hateful": 1, "neutral": 2}).tolist()

# Evaluate both models
def evaluate_model(model_name):
    if model_name == "mbert":
        tokenizer = BertTokenizer.from_pretrained("models/fine_tuned/mbert")
        model = BertForSequenceClassification.from_pretrained("models/fine_tuned/mbert")
    else:
        tokenizer = XLMRobertaTokenizer.from_pretrained("models/fine_tuned/xlm-r")
        model = XLMRobertaForSequenceClassification.from_pretrained("models/fine_tuned/xlm-r")
    
    inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    preds = torch.softmax(outputs.logits, dim=-1).argmax(dim=1).tolist()
    return classification_report(test_labels, preds, output_dict=True)

# Generate comparison plot
mbert_report = evaluate_model("mbert")
xlmr_report = evaluate_model("xlm-r")

metrics = ["precision", "recall", "f1-score"]
models = ["mBERT", "XLM-R"]

fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(metrics))

for i, model in enumerate(models):
    scores = [mbert_report["weighted avg"][m] if model == "mBERT" else xlmr_report["weighted avg"][m] for m in metrics]
    ax.bar(index + i * bar_width, scores, bar_width, label=model)

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison (mBERT vs. XLM-R)')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics)
ax.legend()
plt.show()