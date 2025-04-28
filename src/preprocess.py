import pandas as pd
import os

# Helper: load tsv, csv, or excel automatically
def load_file(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    elif ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Helper: find the correct label column
def find_label_column(df):
    possible_cols = ["task1", "task_1", "labels", "label"]
    for col in possible_cols:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found! Available columns: {list(df.columns)}")

# Helper: process one dataframe
def process_dataframe(df, language, label_map):
    text_col = "text"
    label_col = find_label_column(df)

    df = df[[text_col, label_col]].dropna()
    df["label"] = df[label_col].map(label_map)
    df["language"] = language
    df = df.dropna(subset=["label"])
    return df[["text", "label", "language"]]

def preprocess_data(output_file):
    base_dir = "data/raw/HASOC2021"
    datasets = []
    
    label_map_task1 = {"HOF": 1, "NOT": 0, "HOF_OFFENSIVE": 1, "HOF_HATE": 1, "NOT_OFFENSIVE": 0}

    # English files
    english_files = [
        "english_2019_1.tsv",
        "english_2019_2.tsv",
        "english_2020.xlsx",
        "english_2021.csv",
    ]
    for file in english_files:
        print(f"Loading English: {file}")
        df = load_file(os.path.join(base_dir, file))
        datasets.append(process_dataframe(df, language="english", label_map=label_map_task1))
    
    # Hindi files
    hindi_files = [
        "hindi_2019_1.tsv",
        "hindi_2019_2.tsv",
        "hindi_2020.xlsx",
        "hindi_2021.csv",
    ]
    for file in hindi_files:
        print(f"Loading Hindi: {file}")
        df = load_file(os.path.join(base_dir, file))
        datasets.append(process_dataframe(df, language="hindi", label_map=label_map_task1))
    
    # Marathi file
    print("Loading Marathi: mr_Hasoc2021_train.xlsx")
    df_marathi = load_file(os.path.join(base_dir, "mr_Hasoc2021_train.xlsx"))
    datasets.append(process_dataframe(df_marathi, language="marathi", label_map=label_map_task1))

    # Combine all
    full_df = pd.concat(datasets, ignore_index=True)
    print(f"Total samples combined: {len(full_df)}")
    
    # Save
    print(f"Saving to {output_file}...")
    full_df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    output_file = "data/processed/hasoc2021_multilingual.csv"
    preprocess_data(output_file)
