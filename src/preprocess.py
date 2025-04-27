import json
import pandas as pd
from collections import Counter
from gan import TextGAN  # Import the GAN class
from transformers import BertTokenizer

# Load dataset
def load_dataset(file_path):
    """Load the dataset from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Process dataset and aggregate labels
def process_dataset(data):
    """Process the dataset and aggregate labels using majority voting."""
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
    return pd.DataFrame(processed)

# Generate synthetic data using GAN
def generate_synthetic_data(gan, tokenizer, num_samples=5000, vocab_size=1000):
    """Generate synthetic data using a GAN and decode it into text."""
    synthetic_data = gan.generate_synthetic_data(
        num_samples=num_samples,
        start_token=0,
        vocab_size=vocab_size
    )
    
    # Decode synthetic data back to text
    decoded_synthetic = [
        {"text": tokenizer.decode(seq, skip_special_tokens=True), "label": "hateful"}
        for seq in synthetic_data
    ]
    return pd.DataFrame(decoded_synthetic)

# Main preprocessing function
def preprocess_data(input_file, output_file, num_synthetic_samples=5000):
    """Preprocess the dataset by combining real and synthetic data."""
    # Load and process real dataset
    print("Loading and processing real dataset...")
    data = load_dataset(input_file)
    real_df = process_dataset(data)

    # Initialize GAN and tokenizer
    print("Initializing GAN and tokenizer...")
    gan = TextGAN(vocab_size=1000, embedding_dim=128, hidden_dim=256, max_seq_len=50)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Generate synthetic data
    print(f"Generating {num_synthetic_samples} synthetic samples...")
    synthetic_df = generate_synthetic_data(gan, tokenizer, num_samples=num_synthetic_samples, vocab_size=1000)

    # Combine real and synthetic data
    print("Combining real and synthetic data...")
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)

    # Save preprocessed data
    combined_df.to_csv(output_file, index=False)
    print(f"Dataset preprocessed and saved as '{output_file}'")

# Run preprocessing
if __name__ == "__main__":
    preprocess_data(
        input_file="data/raw/HateXplain/data/dataset.json",
        output_file="data/processed/hate_speech_augmented.csv",
        num_synthetic_samples=5000  # Adjust the number of synthetic samples as needed
    )