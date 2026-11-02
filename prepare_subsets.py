import pandas as pd

# Original full dataset size
TOTAL_DATASET_SIZE = 4838

# Load the existing 70% training set
train_path = "data/processed/train.csv"
df = pd.read_csv(train_path)

print(f"Original train.csv samples: {len(df)}")
print(f"Original full dataset samples: {TOTAL_DATASET_SIZE}")

# Percentages refer to the ORIGINAL 4838-sample dataset
percentages = [3,6]

for percent in percentages:

    # Number of samples based on the FULL dataset
    n_samples = round(TOTAL_DATASET_SIZE * percent / 100)

    # 70% should use the complete original train.csv
    n_samples = min(n_samples, len(df))

    # Keep the ORIGINAL ORDER — no random sampling
    subset = df.iloc[:n_samples].copy()

    output_path = f"data/processed/train_{percent}.csv"
    subset.to_csv(output_path, index=False)

    print(f"\n========== TRAIN {percent}% ==========")
    print(f"Samples: {len(subset)}")
    print("Label distribution:")
    print(subset["label"].value_counts())
    print(f"Saved to: {output_path}")

print("\nAll created!")