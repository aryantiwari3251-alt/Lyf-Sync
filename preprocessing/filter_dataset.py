import pandas as pd

CSV_PATH = "data/raw/Data_Entry_2017.csv"
OUTPUT_PATH = "data/processed/filtered_dataset.csv"

TARGET_DISEASES = {
    "No Finding",
    "Pneumonia",
    "Pneumothorax",
    "Cardiomegaly",
    "Effusion",
    "Atelectasis"
}
KEEP_MULTILABEL = False

df = pd.read_csv(CSV_PATH)

print(f"Original Dataset Size : {len(df)}")

filtered_rows = []

for _, row in df.iterrows():

    labels = row["Finding Labels"].split("|")

    # Keep only labels that belong to target diseases
    valid_labels = [d for d in labels if d in TARGET_DISEASES]

    # Skip if no desired disease exists
    if len(valid_labels) == 0:
        continue

    # ------------------------------
    # Single Label Dataset
    # ------------------------------
    if not KEEP_MULTILABEL:

        if len(labels) != 1:
            continue

        if labels[0] not in TARGET_DISEASES:
            continue

        row["Finding Labels"] = labels[0]
        filtered_rows.append(row)

    # ------------------------------
    # Multi Label Dataset
    # ------------------------------
    else:

        row["Finding Labels"] = "|".join(valid_labels)
        filtered_rows.append(row)


filtered_df = pd.DataFrame(filtered_rows)

print(f"Filtered Dataset Size : {len(filtered_df)}")

print("\nDisease Distribution\n")
print(filtered_df["Finding Labels"].value_counts())

filtered_df.to_csv(OUTPUT_PATH, index=False)

print("\nFiltered dataset saved successfully!")


df["Num_Labels"] = df["Finding Labels"].apply(lambda x: len(x.split("|")))
single_label_df = df[df["Num_Labels"] == 1].copy()

TARGET_DISEASES = [
    "No Finding",
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Pneumothorax",
    "Cardiomegaly",
    "Pneumonia"
]
filtered_df = single_label_df[
    single_label_df["Finding Labels"].isin(TARGET_DISEASES)
].copy()


print(filtered_df["Finding Labels"].value_counts())

