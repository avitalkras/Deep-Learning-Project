# Dataset Download Instructions

## Overview
This script downloads the WikiArt dataset from Kaggle and filters it to contain only the **Post_Impressionism** folder to save space.

## How to Use in Kaggle

### Step 1: Create a New Kaggle Notebook
1. Go to [Kaggle.com](https://www.kaggle.com)
2. Click "New Notebook"
3. Make sure the **WikiArt dataset is connected**:
   - Click "Add Data" (top right)
   - Search for "wikiart" by steubk
   - Click "Add" to connect it

### Step 2: Copy the Script
1. Open `download_dataset.py` from this repository
2. Copy all the code

### Step 3: Run in Kaggle Notebook
1. In your Kaggle notebook, create a new code cell
2. Paste the code from `download_dataset.py`
3. Run the cell

**OR** you can import it directly:

```python
# In Kaggle notebook, upload the file or copy-paste the function
exec(open('download_dataset.py').read())

# Then run:
download_and_filter_dataset()
```

### Step 4: Wait for Download
- The download will take **30-60 minutes** (32GB dataset)
- **DO NOT interrupt** the download
- The script will automatically:
  1. Download the full dataset
  2. Extract only Post_Impressionism folder
  3. Clean up the zip file
  4. Verify the result

### Step 5: Use the Dataset
After completion, the Post_Impressionism folder will be at:
```
/kaggle/working/Post_Impressionism
```

You can then use it in your notebook:
```python
import os
import pandas as pd

base_dir = "/kaggle/working/Post_Impressionism"

records = []
for fname in os.listdir(base_dir):
    if not fname.lower().endswith((".jpg", ".png")):
        continue
    artist = fname.split("_")[0]
    records.append({
        "filepath": os.path.join(base_dir, fname),
        "artist": artist,
        "is_van_gogh": int("van-gogh" in artist.lower())
    })

df = pd.DataFrame(records)
print(f"Total images: {len(df)}")
```

## How to Use in Google Colab

If you want to use this in Colab instead:

1. Upload `download_dataset.py` to Colab
2. Modify the `data_root` path in the script:
   ```python
   data_root = "/content/data"  # Change from /kaggle/working
   ```
3. Make sure Kaggle API is configured (kaggle.json)
4. Run the script

## Output

After successful completion, you'll see:
- ✓ Post_Impressionism folder ready!
- Location: `/kaggle/working/Post_Impressionism`
- Number of images
- Total size (~2-3GB instead of 32GB)

## Troubleshooting

### Download Interrupted
If the download is interrupted:
- Run the script again
- It will check if the zip file exists and resume if possible
- If zip is incomplete, delete it and start over

### No Zip File Found
- Make sure Kaggle API is configured
- Check your internet connection
- Verify the dataset name is correct: `steubk/wikiart`

### Extraction Fails
- Make sure you have enough disk space
- Check that the zip file is complete (should be ~32GB)
- Try running the extraction step separately

## Notes

- The script saves ~30GB by only keeping Post_Impressionism
- Full dataset is ~32GB, filtered folder is ~2-3GB
- The download must complete fully before extraction
- In Kaggle, the dataset will persist in `/kaggle/working/` until the session ends

