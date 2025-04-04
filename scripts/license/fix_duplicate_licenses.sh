#!/bin/bash

# Script to find and fix files with duplicate license headers

# Function to check and fix a file
check_and_fix_file() {
    local file=$1
    
    # Count occurrences of license header
    license_count=$(grep -c "Proprietary Software License Version 1.0" "$file")
    
    if [ "$license_count" -gt 1 ]; then
        echo "Found $license_count license headers in $file, fixing..."
        
        # Create a temporary file
        local tmp_file=$(mktemp)
        
        # Extract just the first license header
        license_header=$(grep -A 6 -m 1 "Proprietary Software License Version 1.0" "$file")
        
        # Save all non-license content to a temp file (skip all license headers)
        grep -v -A 6 "Proprietary Software License Version 1.0" "$file" > "$tmp_file" || true
        
        # Create the fixed file with one license header at the top
        echo "$license_header" > "$file"
        echo "" >> "$file"  # Add a blank line after license
        
        # Add the non-license content, starting after any empty lines or comments at the beginning
        # First, find where real code starts (imports, class definitions, etc.)
        start_line=$(grep -n -m 1 -E "
^
import|
^
@|
^c
lass|
^
struct|
^
enum|
^
protocol|
^
extension|
^
func|
^
var|
^
let" "$tmp_file" | cut -d: -f1)
        
        if [ -n "$start_line" ]; then
            # Extract the code starting from the import or other code element
            tail -n +$start_line "$tmp_file" >> "$file"
        else
            # If no code markers found, just add the entire content
            cat "$tmp_file" >> "$file"
        fi
        
        rm "$tmp_file"
        
        git add "$file"
        echo "  ✓ Fixed license headers in $file"
        return 0
    fi
    
    return 1  # No duplicates found
}

# Fix the HomeViewControllerExtensions.swift file which we know has issues
fixed_count=0
check_and_fix_file "iOS/Views/Home/HomeViewControllerExtensions.swift" && fixed_count=$((fixed_count + 1))

# Find and check all other Swift files
echo "Scanning all files for duplicate license headers..."
find . -type f -name "*.swift" -o -name "*.h" -o -name "*.mm" -o -name "*.cpp" -o -name "*.hpp" | while read file; do
    if [ "$file" != "iOS/Views/Home/HomeViewControllerExtensions.swift" ]; then
        check_and_fix_file "$file" && fixed_count=$((fixed_count + 1))
    fi
done

echo "Fixed $fixed_count files with duplicate license headers"

# If any files were fixed, commit the changes
if [ "$fixed_count" -gt 0 ]; then
    git commit -m "Fix duplicate license headers in $fixed_count files"
    echo "Changes committed. Run 'git push' to push the changes."
else
    echo "No files needed fixing"
fi
