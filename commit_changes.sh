#!/bin/bash
# Check if changes work
echo "Testing model download functionality..."
python test_model_download.py

# Add all files to git
git add utils/model_download.py
git add config.py
git add app.py
git add requirements.txt
git add models/README.md
git add render.yaml
git add test_model_download.py

# Create a commit
git commit -m "Add Google Drive integration for automatic base model download"

echo "Changes committed and ready to push"
