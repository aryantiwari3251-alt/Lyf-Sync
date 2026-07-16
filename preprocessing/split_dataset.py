import pandas as pd
from sklearn.model_selection import train_test_split

# ============================
# Load Filtered Dataset
# ============================

df = pd.read_csv("data/processed/filtered_dataset.csv")

print(f"Total Samples : {len(df)}")

# ============================
# Train (70%) / Temp (30%)
# ============================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["Finding Labels"],
    random_state=42
)

# ============================
# Validation (15%) / Test (15%)
# ============================

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["Finding Labels"],
    random_state=42
)

# ============================
# Save
# ============================

train_df.to_csv("data/processed/train.csv", index=False)
valid_df.to_csv("data/processed/valid.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nDataset Split Complete\n")

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(valid_df)}")
print(f"Test       : {len(test_df)}")