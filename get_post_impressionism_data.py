"""
Simple script to extract Post_Impressionism data from Kaggle WikiArt dataset
and save it as CSV file.

Usage:
- In Kaggle: Dataset already connected, just run this script
- In Colab: Make sure dataset is downloaded first
"""

import os
import pandas as pd

# Find Post_Impressionism folder (works in both Kaggle and Colab)
if os.path.exists("/kaggle/input/wikiart/Post_Impressionism"):
    base_dir = "/kaggle/input/wikiart/Post_Impressionism"
    output_file = "/kaggle/working/post_impressionism_data.csv"
elif os.path.exists("/content/data/Post_Impressionism"):
    base_dir = "/content/data/Post_Impressionism"
    output_file = "/content/post_impressionism_data.csv"
else:
    raise FileNotFoundError("Post_Impressionism folder not found!")

# Create DataFrame
print("Reading Post_Impressionism images...")
records = []

for fname in os.listdir(base_dir):
    if not fname.lower().endswith((".jpg", ".png")):
        continue
    
    artist = fname.split("_")[0]  # Extract artist name from filename
    
    records.append({
        "filepath": os.path.join(base_dir, fname),
        "filename": fname,
        "artist": artist,
        "is_van_gogh": 1 if "van-gogh" in artist.lower() else 0
    })

df = pd.DataFrame(records)

# Save to CSV
df.to_csv(output_file, index=False)

print(f"\n✅ Done!")
print(f"CSV file saved: {output_file}")
print(f"Total images: {len(df)}")
print(f"Van Gogh: {df['is_van_gogh'].sum()}")
print(f"Other: {len(df) - df['is_van_gogh'].sum()}")
