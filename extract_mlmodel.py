import os
import sys
from pathlib import Path
import shutil

# Try to import coremltools, handle if not installed
try:
    import coremltools as ct
except ImportError:
    print("Error: coremltools is not installed. Please install it with 'pip install coremltools'")
    sys.exit(1)

def extract_mlmodel_info(mlmodel_path, output_dir):
    """Extract information from the .mlmodel file."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(mlmodel_path):
        print(f"Error: File {mlmodel_path} not found")
        sys.exit(1)

    try:
        # Load the Core ML model
        model = ct.models.MLModel(mlmodel_path)
        
        # Extract model specification
        spec = model.get_spec()
        
        # Save the full specification as a text file
        with open(os.path.join(output_dir, "model_spec.txt"), "w") as f:
            f.write(str(spec))
        
        # Extract and save input description
        with open(os.path.join(output_dir, "inputs.txt"), "w") as f:
            for input in spec.description.input:
                f.write(f"Input Name: {input.name}\n")
                f.write(f"Type: {input.type}\n\n")
        
        # Extract and save output description
        with open(os.path.join(output_dir, "outputs.txt"), "w") as f:
            for output in spec.description.output:
                f.write(f"Output Name: {output.name}\n")
                f.write(f"Type: {output.type}\n\n")
        
        # If it's a neural network, extract layer info
        if spec.WhichOneof('Type') == 'neuralNetwork':
            with open(os.path.join(output_dir, "layers.txt"), "w") as f:
                for i, layer in enumerate(spec.neuralNetwork.layers):
                    f.write(f"Layer {i}: {layer.name}\n")
                    f.write(f"Type: {layer.WhichOneof('layer')}\n\n")
        
        print(f"Extracted model information to {output_dir}")
        
    except Exception as e:
        print(f"Error processing .mlmodel file: {str(e)}")
        # As a fallback, copy the original file
        shutil.copy(mlmodel_path, output_dir)
        print(f"Copied raw .mlmodel file to {output_dir} as fallback")

def create_zip_from_directory(directory, zip_name):
    """Create a ZIP file from a directory."""
    shutil.make_archive(zip_name.replace('.zip', ''), 'zip', directory)
    return f"{zip_name}"

def main():
    ML_MODEL_PATH = "model/coreml_model.mlmodel"
    OUTPUT_DIR = "extracted_model"
    ZIP_NAME = "model_contents.zip"
    
    # Extract model information
    extract_mlmodel_info(ML_MODEL_PATH, OUTPUT_DIR)
    
    # Create a ZIP of the extracted contents
    zip_path = create_zip_from_directory(OUTPUT_DIR, ZIP_NAME)
    print(f"Created ZIP file: {zip_path}")

if __name__ == "__main__":
    main()
