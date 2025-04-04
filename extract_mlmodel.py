import os
import zipfile
import shutil
import sys
from pathlib import Path

def is_zip_file(filepath):
    """Check if the file is a ZIP archive."""
    try:
        with open(filepath, 'rb') as f:
            return f.read(4) == b'PK\x03\x04'  # ZIP file magic number
    except Exception:
        return False

def extract_mlmodel(mlmodel_path, output_dir):
    """Extract contents of mlmodel file or copy it if it's not a ZIP."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(mlmodel_path):
        print(f"Error: File {mlmodel_path} not found")
        sys.exit(1)

    if is_zip_file(mlmodel_path):
        try:
            with zipfile.ZipFile(mlmodel_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                print(f"Extracted contents to {output_dir}")
        except zipfile.BadZipFile:
            print("Warning: Not a valid ZIP file, copying raw file instead")
            shutil.copy(mlmodel_path, output_dir)
    else:
        print("File is not a ZIP archive, copying raw file")
        shutil.copy(mlmodel_path, output_dir)

def create_zip_from_directory(directory, zip_name):
    """Create a ZIP file from a directory."""
    shutil.make_archive(zip_name.replace('.zip', ''), 'zip', directory)
    return f"{zip_name}"

def main():
    # Configuration
    ML_MODEL_PATH = "model/coreml_model.mlmodel"  # Path to your .mlmodel file
    OUTPUT_DIR = "extracted_model"                # Where to extract contents
    ZIP_NAME = "model_contents.zip"              # Name of the output ZIP
    
    # Extract the mlmodel file
    extract_mlmodel(ML_MODEL_PATH, OUTPUT_DIR)
    
    # Create a ZIP of the extracted contents
    zip_path = create_zip_from_directory(OUTPUT_DIR, ZIP_NAME)
    print(f"Created ZIP file: {zip_path}")

if __name__ == "__main__":
    main()