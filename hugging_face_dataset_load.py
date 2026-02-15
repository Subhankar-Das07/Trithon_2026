from datasets import load_dataset
import pandas as pd

print("🔄 Loading dataset from Parquet file...")
dataset = load_dataset(
    "parquet",
    data_files="https://huggingface.co/datasets/jglaser/binding_affinity/resolve/main/data/all.parquet",
    split="train"
)
print(f"✅ Loaded {len(dataset)} rows")

df = dataset.to_pandas()
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:", df.columns.tolist())

# Save a small sample for quick loading later (optional)
df_sample = df.sample(frac=0.1, random_state=42)
df_sample.to_csv("sample_10percent.csv", index=False)
print("\n✅ Saved 10% sample to sample_10percent.csv")