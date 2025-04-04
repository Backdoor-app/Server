#!/bin/bash

# Script to handle merge conflicts while preserving license headers

# Function to fix a merge conflict in a file
fix_merge_conflict() {
    local file=$1
    local tmp_file=$(mktemp)
    
    echo "Fixing merge conflict in $file"
    
    # First, check if the file has the license header in our branch
    if grep -q "Proprietary Software License Version 1.0" "$file"; then
        # Extract the license header from our branch
        license_header=$(sed -n '/Proprietary Software License/,/under the terms of the Proprietary Software License/p' "$file")
        
        # Checkout the main branch version to get its content
        git checkout --theirs "$file"
        
        # Now insert the license header at the top of the file
        echo "$license_header" > "$tmp_file"
        echo "" >> "$tmp_file"  # Add a blank line after license
        cat "$file" >> "$tmp_file"
        
        # Remove any duplicate license headers that might have been added
        sed -i '/Proprietary Software License Version 1.0/{n;/Proprietary Software License Version 1.0/,/under the terms of the Proprietary Software License/d;}' "$tmp_file"
        
        # Replace the file with our fixed version
        cp "$tmp_file" "$file"
        rm "$tmp_file"
        
        # Mark as resolved
        git add "$file"
        echo "  ✓ Fixed and staged $file"
    else
        # No license header in our branch, just use main's version
        git checkout --theirs "$file"
        git add "$file"
        echo "  ✓ Used main branch version for $file"
    fi
}

# Start a merge with main
echo "Starting merge with main branch..."
git merge main || true

# Get list of conflicting files
conflicting_files=$(git diff --name-only --diff-filter=U)

if [ -z "$conflicting_files" ]; then
    echo "No merge conflicts found."
    exit 0
fi

echo "Found conflicts in the following files:"
echo "$conflicting_files"
echo ""

# Process each conflicting file
echo "$conflicting_files" | while read file; do
    if [ -f "$file" ]; then
        fix_merge_conflict "$file"
    fi
done

# Check if all conflicts are resolved
if [ -z "$(git diff --name-only --diff-filter=U)" ]; then
    echo "All conflicts resolved. Completing the merge..."
    git commit -m "Merge main branch while preserving license headers"
    echo "Merge completed successfully!"
else
    echo "Some conflicts remain. Please resolve them manually."
    git diff --name-only --diff-filter=U
fi
