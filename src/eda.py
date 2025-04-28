# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv("data/processed/hasoc2021_multilingual.csv")

# Show basic stats
print(f"Total Samples: {len(df)}")
print("\nSamples per Language:")
print(df['language'].value_counts())

print("\nClass Distribution (0=Non-Offensive, 1=Offensive):")
print(df['label'].value_counts())

# Plot 1: Samples per Language
plt.figure(figsize=(8,4))
sns.countplot(data=df, x='language', order=df['language'].value_counts().index, palette='viridis')
plt.title('Samples per Language')
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_language_distribution.png')
plt.close()

# Plot 2: Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='label', palette='magma')
plt.title('Class Distribution (0=Non-Offensive, 1=Offensive)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda_label_distribution.png')
plt.close()

print("\nEDA plots saved: eda_language_distribution.png, eda_label_distribution.png")
