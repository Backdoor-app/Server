# Backdoor AI Model Files

This directory contains model files and metadata for the Backdoor AI server.

## Base CoreML Model

**Important**: The base CoreML model is not included in this repository because it exceeds GitHub's file size limit (459MB).

### How to get the model

1. **Option 1: Use the GitHub workflow**
   - The GitHub workflow in `.github/workflows/main.yml` automatically downloads the model from Dropbox
   - Run the workflow manually or push to trigger it

2. **Option 2: Download directly**
   - Download from the Dropbox URL in the GitHub workflow file:
   ```
   https://www.dropbox.com/scl/fi/2xarhyii46tr9amkqh764/coreml_model.mlmodel?rlkey=j3cxmpjhxj8bbwzw11j1hy54c&st=zuyjx83u&dl=1
   ```

3. **Option 3: Use your own model**
   - If you have your own CoreML model for intent classification, you can use it instead

### Where to place the model

Place the downloaded model in the following locations:
- `model/coreml_model.mlmodel` (original location used by GitHub workflow)
- `models/model_1.0.0.mlmodel` (location used by the server)

## Model Metadata

The model metadata files (`latest_model.json` and `model_info_1.0.0.json`) are included in this repository. These files contain information about the model version, accuracy, and structure.

## Uploaded Models Directory

The `uploaded/` directory is where user-uploaded models will be stored when the server is running.
