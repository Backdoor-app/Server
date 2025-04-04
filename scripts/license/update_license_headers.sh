# Proprietary Software License Version 1.0
#
# Copyright (C) 2025 BDG
#
# Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.

#!/bin/bash

#!/bin/bash
#
# Proprietary Software License Version 1.0
#
# Copyright (C) 2025 BDG
#
# Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
#


# Script to remove all comments at the top of files and add a proprietary license header
# Created for Backdoor App Signer

# Define the license text for different file types
SWIFT_HEADER="//
//"

CPP_HEADER="/*
 */"

SHELL_HEADER="#!/bin/bash
#
#"

# Create a temporary file to track processed files
PROCESSED_FILES_LIST=$(mktemp)
echo "0" > "$PROCESSED_FILES_LIST"

# Function to process a file
process_file() {
    local file=$1
    local header=$2
    local keep_shebang=$3
    local tmp_file=$(mktemp)
    local is_modified=false
    
    echo "Processing $file..."
    
    # Skip files containing "LICENSE" in the filename (for safety)
    if [[ "$(basename "$file")" == *LICENSE* ]]; then
        echo "  Skipping license file: $file"
        return
    fi
    
    # Skip this script itself
    if [[ "$(basename "$file")" == "update_license_headers.sh" ]]; then
        echo "  Skipping this script itself"
        return
    fi
    
    # Determine if it's a text file
    if ! file "$file" | grep -q "text"; then
        echo "  Skipping non-text file: $file"
        return
    fi
    
    # Detect if file has shebang and needs to preserve it
    if [[ "$keep_shebang" == "true" ]] && head -n 1 "$file" | grep -q "^#!"; then
        local shebang=$(head -n 1 "$file")
        echo "$shebang" > "$tmp_file"
        echo "" >> "$tmp_file"
        echo "$header" >> "$tmp_file"
        echo "" >> "$tmp_file"
        
        # Find the start of actual code (after comments)
        local code_start_line=1
        if grep -q "^#!" "$file"; then
            # Skip the shebang line when looking for code start
            code_start_line=2
        fi
        
        # Skip any existing license header (look for meaningful code)
        # Find the first line that's not a comment, copyright notice, license mention, or empty
        local first_code_line=$(tail -n +$code_start_line "$file" | grep -n -m 1 -E '^[^#[:space:]]|^[[:space:]]+[^#]' | grep -v -E 'Copyright|License|Proprietary' | cut -d: -f1)
        
        # If no code line found, use the whole file after shebang
        if [ -z "$first_code_line" ]; then
            tail -n +2 "$file" | grep -v -E 'Copyright|License|Proprietary' >> "$tmp_file"
        else
            # Calculate the actual line number in the original file
            local actual_line=$((code_start_line + first_code_line - 1))
            tail -n +$actual_line "$file" >> "$tmp_file"
        fi
        
        is_modified=true
    else
        # No shebang to preserve, just add the header
        echo "$header" > "$tmp_file"
        echo "" >> "$tmp_file"  # Add a blank line after header
        
        # Skip any existing license header at the top
        if [[ "$file" == *.swift ]]; then
            # For Swift files, find the first import or other non-comment declaration
            local first_line=$(grep -n -m 1 -E '^import|^@|^public|^private|^internal|^open|^final|^class|^struct|^enum|^protocol|^extension|^func|^var|^let' "$file" | cut -d: -f1)
            
            # If no marker found, look for any line that's not a comment or blank
            if [ -z "$first_line" ]; then
                first_line=$(grep -n -m 1 -v -E '^[[:space:]]*//(.*)|^[[:space:]]*$|^[[:space:]]*/\*|\*/[[:space:]]*$' "$file" | cut -d: -f1)
            fi
            
            # If we found a line, use it
            if [ -n "$first_line" ]; then
                tail -n +$first_line "$file" >> "$tmp_file"
                is_modified=true
            else
                # No code found, just use the whole file content (unusual case)
                cat "$file" >> "$tmp_file"
                is_modified=true
            fi
            
        elif [[ "$file" == *.cpp || "$file" == *.hpp || "$file" == *.h || "$file" == *.mm ]]; then
            # For C/C++/ObjC files, find the first include, define, or declaration
            local first_line=$(grep -n -m 1 -E '^#include|^#import|^#define|^#pragma|^#ifndef|^#if|^using|^namespace|^class|^struct|^enum|^void|^int|^char|^bool|^template|^typedef' "$file" | cut -d: -f1)
            
            # If no marker found, look for any line that's not a comment or blank
            if [ -z "$first_line" ]; then
                first_line=$(grep -n -m 1 -v -E '^[[:space:]]*//(.*)|^[[:space:]]*$|^[[:space:]]*/\*|\*/[[:space:]]*$' "$file" | cut -d: -f1)
            fi
            
            # If we found a line, use it
            if [ -n "$first_line" ]; then
                tail -n +$first_line "$file" >> "$tmp_file"
                is_modified=true
            else
                # No code found, just use the whole file content (unusual case)
                cat "$file" >> "$tmp_file"
                is_modified=true
            fi
        else
            # For other file types, just find the first non-comment line
            local first_line=$(grep -n -m 1 -v -E '^[[:space:]]*(//|/\*|#)' "$file" | cut -d: -f1)
            
            # If we found a line, use it
            if [ -n "$first_line" ]; then
                tail -n +$first_line "$file" >> "$tmp_file"
                is_modified=true
            else
                # No code found, just use the whole file content (unusual case)
                cat "$file" >> "$tmp_file"
                is_modified=true
            fi
        fi
    fi
    
    # Only replace the file if we actually made changes
    if [ "$is_modified" = true ]; then
        mv "$tmp_file" "$file"
        # Increment the processed files counter in our tracking file
        CURRENT_COUNT=$(cat "$PROCESSED_FILES_LIST")
        echo $((CURRENT_COUNT + 1)) > "$PROCESSED_FILES_LIST"
        echo "  Updated license header in $file"
    else
        rm "$tmp_file"
    fi
}

# Function to process a directory
process_directory() {
    local dir=$1
    echo "Scanning directory: $dir"
    
    # Process Swift files
    find "$dir" -type f -name "*.swift" | while read file; do
        process_file "$file" "$SWIFT_HEADER" "false"
    done
    
    # Process Objective-C header files
    find "$dir" -type f -name "*.h" | while read file; do
        process_file "$file" "$CPP_HEADER" "false"
    done
    
    # Process Objective-C++ files
    find "$dir" -type f -name "*.mm" | while read file; do
        process_file "$file" "$CPP_HEADER" "false"
    done
    
    # Process C++ files
    find "$dir" -type f -name "*.cpp" | while read file; do
        process_file "$file" "$CPP_HEADER" "false"
    done
    
    # Process C++ header files
    find "$dir" -type f -name "*.hpp" | while read file; do
        process_file "$file" "$CPP_HEADER" "false"
    done
    
    # Process Shell scripts (preserving shebang)
    find "$dir" -type f -name "*.sh" | grep -v "update_license_headers.sh" | while read file; do
        process_file "$file" "$SHELL_HEADER" "true"
    done
}

# Main execution
echo "Starting license header update process..."

# Process main directories
process_directory "iOS"
process_directory "Shared"

# Process root shell scripts
find . -maxdepth 1 -type f -name "*.sh" | grep -v "update_license_headers.sh" | while read file; do
    process_file "$file" "$SHELL_HEADER" "true"
done

# Get the final count from our tracking file
FINAL_COUNT=$(cat "$PROCESSED_FILES_LIST")
echo "Completed! $FINAL_COUNT files have been updated with the new license header."

# Clean up temp file
rm "$PROCESSED_FILES_LIST"
