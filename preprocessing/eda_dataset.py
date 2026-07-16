#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# In[2]:


csv_path = "data/raw/Data_Entry_2017.csv"

import os

print("Current Directory:", os.getcwd())
print("CSV Path:", csv_path)
print("Absolute Path:", os.path.abspath(csv_path))
print("Exists:", os.path.exists(csv_path))

df = pd.read_csv(csv_path)
OUTPUT_PATH = "data/analysis/filtered_dataset.csv"
df = pd.read_csv(csv_path)

# print("=" * 50)
print("Dataset Shape")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nMissing Values")
print(df.isnull().sum())


print("\nDuplicate Rows :", df.duplicated().sum())


all_labels = []

for labels in df["Finding Labels"]:
    diseases = labels.split("|")
    all_labels.extend(diseases)

disease_counter = Counter(all_labels)

disease_df = (
    pd.DataFrame(
        disease_counter.items(),
        columns=["Disease", "Count"]
    )
    .sort_values("Count", ascending=False)
    .reset_index(drop=True)
)

print("\nDisease Distribution")
print(disease_df)


df["Num_Labels"] = df["Finding Labels"].apply(
    lambda x: len(x.split("|"))
)
label_count = (
    df["Num_Labels"]
    .value_counts()
    .sort_index()
)

print("\nNumber of Diseases per Image")
print(label_count)


print("\nView Position Distribution")

print(df["View Position"].value_counts())


print("\nGender Distribution")

print(df["Patient Gender"].value_counts())


print("\nPatient Age Statistics")
print(df["Patient Age"].describe())


plt.figure(figsize=(12,6))

plt.bar(
    disease_df["Disease"],
    disease_df["Count"]
)

plt.xticks(rotation=45)

plt.xlabel("Disease")

plt.ylabel("Number of Images")

plt.title("Disease Distribution")

plt.tight_layout()

plt.savefig("disease_distribution.png")

plt.show()


plt.figure(figsize=(6,4))

plt.bar(
    label_count.index.astype(str),
    label_count.values
)
plt.xlabel("Number of Labels")

plt.ylabel("Images")

plt.title("Multi-label Distribution")

plt.tight_layout()

plt.savefig("multilabel_distribution.png")

plt.show()

print("\nEDA Completed Successfully!")


