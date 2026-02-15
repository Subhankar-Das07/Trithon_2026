# prepare_data.py
import pandas as pd
import os
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from binding_dataset import BindingDataset, collate_fn
import torch

# Try to load sample CSV if it exists
csv_path = "sample_10percent.csv"
if os.path.exists(csv_path):
    print("Loading existing sample CSV...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV.")
else:
    print("Sample CSV not found. Loading dataset from Hugging Face Parquet URL...")
    # Load from the public Parquet URL
    dataset = load_dataset(
        "parquet",
        data_files="https://huggingface.co/datasets/jglaser/binding_affinity/resolve/main/data/all.parquet",
        split="train"
    )
    full_df = dataset.to_pandas()
    print(f"Full dataset loaded: {len(full_df)} rows.")
    
    # Take a 10% sample for development
    df = full_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"Created sample with {len(df)} rows.")
    
    # Save sample to CSV for future runs
    df.to_csv(csv_path, index=False)
    print(f"Sample saved to {csv_path}.")

# Create dataset instance
dataset = BindingDataset(df)

# Split into train/val/test (80/10/10)
total = len(dataset)
train_len = int(0.8 * total)
val_len = int(0.1 * total)
test_len = total - train_len - val_len

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Create DataLoaders
batch_size = 32  # adjust based on your memory

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=0, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=collate_fn, num_workers=0, drop_last=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=collate_fn, num_workers=0, drop_last=False
)

print("DataLoaders created successfully.")

# Test one batch to ensure it works
for batch in train_loader:
    if batch[0] is None:
        continue
    graphs, prots, affs = batch
    print(f"Batch graphs: {graphs}")
    print(f"Proteins shape: {prots.shape}")
    print(f"Affinities shape: {affs.shape}")
    break