import zipfile
import os

# The zip is in the same folder as this script
zip_path = 'archive (3).zip' 
extract_path = 'UCSD_Dataset'

if os.path.exists(zip_path):
    print(f"Opening {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Extraction successful! Folder 'UCSD_Dataset' created.")
else:
    print(f"❌ Error: Could not find {zip_path} in this folder.")
    print(f"Current Folder: {os.getcwd()}")