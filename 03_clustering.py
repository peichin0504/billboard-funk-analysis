"""
03_clustering.py
Uses K-Means clustering to classify songs into funk vs. control groups.
Clustering is fitted on pre-2015 data only to avoid look-ahead bias.
Requires: merged_data.csv
Output: classified_data.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load merged data
df = pd.read_csv("merged_data.csv")

# Features used to define funk: high danceability, energy, valence; low acousticness
funk_features = ["danceability", "energy", "valence", "acousticness", "tempo"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df[funk_features])

# Fit clustering on pre-2015 data only to avoid look-ahead bias
pre2015_mask = df["year"] < 2015
X_pre = X[pre2015_mask]

# Elbow method to select optimal k
inertias = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pre)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(2, 8), inertias, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=150)
plt.show()

# Fit final model with k=3 (elbow at k=3)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_pre)

# Assign clusters to all songs
df["cluster"] = km.predict(X)

# Examine cluster centers to identify the funk cluster
centers = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_),
    columns=funk_features
)
print("Cluster centers:")
print(centers.round(3))

# Funk cluster = highest danceability + valence + energy
funk_cluster = centers["danceability"].idxmax()
print(f"\nFunk cluster identified: {funk_cluster}")

# Label funk vs control
df["is_funk"] = (df["cluster"] == funk_cluster).astype(int)
print(f"Funk songs: {df['is_funk'].sum()}")
print(f"Control songs: {(df['is_funk'] == 0).sum()}")

df.to_csv("classified_data.csv", index=False)
print("Saved: classified_data.csv")
