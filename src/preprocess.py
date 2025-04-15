import json
import pandas as pd
from collections import Counter
from gan import TextGAN  # Import the GAN class
from transformers import BertTokenizer

# Load dataset
with open("data/raw/HateXplain/data/dataset.json", "r") as f:
    data = json.load(f)

# Extract text and labels
processed = []
for post_id, post in data.items():
    text = " ".join(post["post_tokens"])
    
    # Aggregate annotator labels
    annotator_labels = [annotator["label"] for annotator in post["annotators"]]
    if not annotator_labels:  # Skip if no labels are available
        continue
    
    label_counts = Counter(annotator_labels)
    final_label = label_counts.most_common(1)[0][0]  # Majority voting
    
    processed.append({"text": text, "label": final_label})

df = pd.DataFrame(processed)

# Generate synthetic hate speech with GAN
gan = TextGAN(vocab_size=1000, embedding_dim=128, hidden_dim=256, max_seq_len=50)
synthetic_data = gan.generate_synthetic_data(
    num_samples=5000,
    start_token=0,
    vocab_size=1000
)

# Debug: Inspect synthetic data
print("Synthetic Data Shape:", len(synthetic_data), len(synthetic_data[0]))

# Decode synthetic data back to text
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
decoded_synthetic = [
    {"text": tokenizer.decode(seq, skip_special_tokens=True), "label": "hateful"}
    for seq in synthetic_data
]

# Combine real and synthetic data
synthetic_df = pd.DataFrame(decoded_synthetic)
df = pd.concat([df, synthetic_df])

# Save preprocessed data
df.to_csv("data/processed/hate_speech_augmented.csv", index=False)
print("Dataset preprocessed and saved as 'hate_speech_dataset.csv'")