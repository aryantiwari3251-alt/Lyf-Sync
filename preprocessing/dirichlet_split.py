import numpy as np
import pandas as pd

# ==========================================
# Configuration
# ==========================================

NUM_CLIENTS = 3
ALPHA = 0.5
SEED = 42

np.random.seed(SEED)

# ==========================================
# Load Training Set
# ==========================================

train_df = pd.read_csv("data/processed/train.csv")

classes = train_df["Finding Labels"].unique()

client_indices = [[] for _ in range(NUM_CLIENTS)]

# ==========================================
# Perform Dirichlet Split
# ==========================================

for disease in classes:

    disease_idx = train_df[
        train_df["Finding Labels"] == disease
    ].index.to_numpy()

    np.random.shuffle(disease_idx)

    proportions = np.random.dirichlet(
        [ALPHA] * NUM_CLIENTS
    )

    split_points = (
        np.cumsum(proportions)[:-1] * len(disease_idx)
    ).astype(int)

    split_indices = np.split(
        disease_idx,
        split_points
    )

    for client_id in range(NUM_CLIENTS):

        client_indices[client_id].extend(
            split_indices[client_id]
        )

# ==========================================
# Save Hospital CSVs
# ==========================================

hospital_names = [
    "data/processed/hospital_A.csv",
    "data/processed/hospital_B.csv",
    "data/processed/hospital_C.csv"
]

for client_id in range(NUM_CLIENTS):

    hospital_df = train_df.loc[
        client_indices[client_id]
    ].sample(frac=1, random_state=SEED)

    hospital_df.to_csv(
        hospital_names[client_id],
        index=False
    )

    print("\n", hospital_names[client_id])

    print(hospital_df["Finding Labels"].value_counts())