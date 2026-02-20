"""
02_merge_data.py
Merges Billboard chart data with Spotify audio features.
Requires: billboard_2010_2020.csv, dataset-of-10s.csv (from Kaggle)
Output: merged_data.csv

Kaggle dataset: https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset
"""

import pandas as pd

SPOTIFY_PATH = "dataset-of-10s.csv"  # update path if needed

# Load datasets
billboard_df = pd.read_csv("billboard_2010_2020.csv")
spotify_df = pd.read_csv(SPOTIFY_PATH)

# Normalize song titles for matching
billboard_df["title_clean"] = billboard_df["title"].str.lower().str.strip()
spotify_df["track_clean"] = spotify_df["track"].str.lower().str.strip()

# Merge on song title
merged_df = billboard_df.merge(
    spotify_df,
    left_on="title_clean",
    right_on="track_clean",
    how="inner"
)

# Keep relevant columns
feature_cols = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness", "liveness"
]
merged_df = merged_df[["year", "rank", "title", "artist_x"] + feature_cols]
merged_df = merged_df.rename(columns={"artist_x": "artist"})

# Drop duplicates (same song matched to multiple Spotify versions)
merged_df = merged_df.drop_duplicates(subset=["title", "year"])

print(f"Matched songs: {len(merged_df)} / {len(billboard_df)} ({len(merged_df)/len(billboard_df):.1%} coverage)")

merged_df.to_csv("merged_data.csv", index=False)
print("Saved: merged_data.csv")
