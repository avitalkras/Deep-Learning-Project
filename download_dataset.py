"""
Download and Filter WikiArt Dataset from Kaggle
================================================

This script downloads the WikiArt dataset from Kaggle and filters it to contain
only the Post_Impressionism folder to save space.

Usage in Kaggle Notebook:
    Option 1: Copy-paste this entire script into a cell and run it
    
    Option 2: Use in a cell like this:
        exec(open('download_dataset.py').read())
        download_and_filter_dataset()
    
    Option 3: For notebook-style, replace the subprocess call with:
        !kaggle datasets download -d steubk/wikiart -p /kaggle/working

Usage in Google Colab:
    - Upload this file to Colab
    - Change data_root to "/content/data" 
    - Make sure Kaggle API is configured
    - Run the script

Requirements:
    - kaggle package installed
    - Kaggle API credentials configured
    - Dataset connected to notebook (for Kaggle)
"""

import os
import zipfile
import shutil
from tqdm import tqdm


def download_and_filter_dataset():
    """
    Download WikiArt dataset from Kaggle and extract only Post_Impressionism folder.
    
    Returns:
        str: Path to the extracted Post_Impressionism folder
    """
    
    # Configuration
    dataset_name = "steubk/wikiart"
    data_root = "/kaggle/working"  # Kaggle working directory
    target_folder = os.path.join(data_root, "Post_Impressionism")
    temp_dir = os.path.join(data_root, "temp_extract")
    
    # Create directories
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check if already extracted
    if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
        files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✓ Post_Impressionism folder already exists with {len(files)} images")
        print(f"  Location: {target_folder}")
        return target_folder
    
    print("=" * 60)
    print("WikiArt Dataset Downloader")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Target: Post_Impressionism folder only")
    print(f"Full dataset size: ~32GB")
    print("=" * 60)
    
    # Step 1: Download the full dataset
    print("\n[Step 1/3] Downloading full dataset from Kaggle...")
    print("⚠️  WARNING: This will download ~32GB. May take 30-60 minutes.")
    print("    DO NOT interrupt the download!\n")
    
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", data_root],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Download command completed")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e.stderr}")
        raise
    except KeyboardInterrupt:
        print("\n\n⚠️  Download was interrupted (Ctrl+C)")
        print("    Please run this script again to restart the download.")
        raise
    except Exception as e:
        print(f"❌ Download error: {e}")
        raise
    
    # Verify download completed
    zip_files = [f for f in os.listdir(data_root) if f.endswith('.zip')]
    if not zip_files:
        print("\n⚠️  Download was interrupted or incomplete!")
        print("    Please run this script again and let it complete.")
        raise FileNotFoundError(
            "Download incomplete. No zip file found.\n"
            "If you interrupted the download, please run again."
        )
    
    zip_path = os.path.join(data_root, zip_files[0])
    
    # Verify zip file size
    zip_size = os.path.getsize(zip_path) / (1024**3)  # Size in GB
    if zip_size < 1:
        print(f"\n⚠️  Warning: Zip file seems too small ({zip_size:.2f} GB)")
        print("    Expected ~32GB. Download may be incomplete.")
        raise ValueError(f"Zip file appears incomplete (size: {zip_size:.2f} GB)")
    
    print(f"✓ Download completed!")
    print(f"  Zip file: {zip_files[0]}")
    print(f"  Size: {zip_size:.2f} GB")
    
    # Step 2: Extract ONLY Post_Impressionism folder
    print("\n[Step 2/3] Extracting Post_Impressionism folder from zip...")
    print("    This may take 5-10 minutes...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of all files in zip
            print("  Reading zip file contents...")
            all_files = zip_ref.namelist()
            
            # Filter for Post_Impressionism files only
            post_imp_files = [f for f in all_files if 'Post_Impressionism' in f]
            
            if not post_imp_files:
                raise ValueError("Post_Impressionism folder not found in dataset!")
            
            print(f"  Found {len(post_imp_files)} files in Post_Impressionism folder")
            
            # Extract only Post_Impressionism files
            print("  Extracting files...")
            for file in tqdm(post_imp_files, desc="Extracting", unit="files"):
                zip_ref.extract(file, temp_dir)
            
            print("  ✓ Extraction completed!")
            
    except Exception as e:
        print(f"  ❌ Extraction error: {e}")
        raise
    
    # Step 3: Move Post_Impressionism folder to final location
    print("\n[Step 3/3] Organizing files...")
    
    # Find the Post_Impressionism folder in temp directory
    def find_post_imp_folder(root_dir):
        for root, dirs, files in os.walk(root_dir):
            if 'Post_Impressionism' in dirs:
                return os.path.join(root, 'Post_Impressionism')
        return None
    
    extracted_path = find_post_imp_folder(temp_dir)
    
    if extracted_path and os.path.exists(extracted_path):
        # Move to final location
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        shutil.move(extracted_path, target_folder)
        print(f"  ✓ Moved to: {target_folder}")
    else:
        raise FileNotFoundError("Could not find Post_Impressionism folder after extraction")
    
    # Clean up: remove zip file and temp directory
    print("\n🧹 Cleaning up temporary files...")
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("  ✓ Removed zip file (saved ~30GB)")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("  ✓ Removed temporary extraction folder")
    except Exception as e:
        print(f"  ⚠️ Cleanup warning: {e}")
    
    # Verify final result
    print("\n" + "=" * 60)
    print("📊 Final Verification")
    print("=" * 60)
    if os.path.exists(target_folder):
        files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✓ Post_Impressionism folder ready!")
        print(f"  Location: {target_folder}")
        print(f"  Images: {len(files)}")
        
        # Calculate folder size
        total_size = sum(os.path.getsize(os.path.join(target_folder, f)) 
                         for f in os.listdir(target_folder) 
                         if os.path.isfile(os.path.join(target_folder, f)))
        size_gb = total_size / (1024**3)
        print(f"  Size: {size_gb:.2f} GB")
        print("=" * 60)
        
        return target_folder
    else:
        raise FileNotFoundError(f"Post_Impressionism folder not found at {target_folder}")


if __name__ == "__main__":
    print("Starting dataset download and filtering...")
    try:
        result_path = download_and_filter_dataset()
        print(f"\n✅ SUCCESS! Dataset ready at: {result_path}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

