#!/bin/bash

# Final comprehensive cleanup script for license headers

# Define the clean license headers for different file types
SWIFT_HEADER="//
// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
//"

CPP_HEADER="/*
 * Proprietary Software License Version 1.0
 *
 * Copyright (C) 2025 BDG
 *
 * Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
 */"

SHELL_HEADER="# Proprietary Software License Version 1.0
#
# Copyright (C) 2025 BDG
#
# Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License."

# Function to completely clean a file and add a single license header
clean_file() {
    local file=$1
    local header_type=$2
    local tmp_file=$(mktemp)
    
    echo "Cleaning $file..."
    
    # Read the entire file content
    local file_content=$(cat "$file")
    
    # Choose the appropriate header
    local license_header
    case "$header_type" in
        swift) license_header="$SWIFT_HEADER" ;;
        cpp) license_header="$CPP_HEADER" ;;
        shell) license_header="$SHELL_HEADER" ;;
        *) license_header="$SWIFT_HEADER" ;;  # Default to Swift header
    esac
    
    # Remove ALL license headers from the content
    # This uses a more aggressive approach to catch all variations
    file_content=$(echo "$file_content" | sed -E '/Proprietary Software License/,/under the terms of the Proprietary Software License/d')
    
    # Remove merge conflict markers if any
    file_content=$(echo "$file_content" | sed -E '/<<<<<<< HEAD/,/>>>>>>> /d')
    
    # Remove empty lines from the beginning
    file_content=$(echo "$file_content" | sed -E '/
^
[[:space:]]*$/d')
    
    # Find where the actual code starts - look for imports or other code markers
    local code_start=""
    case "$header_type" in
        swift)
            code_start=$(echo "$file_content" | grep -m 1 -E "
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
let")
            ;;
        cpp)
            code_start=$(echo "$file_content" | grep -m 1 -E "
^
#include|
^
#import|
^
#define|
^
#pragma|
^
#ifndef|
^
#if|
^
using|
^
namespace|
^c
lass|
^
struct|
^
enum|
^
void|
^
int|
^c
har|
^
bool|
^
template|
^
typedef")
            ;;
        *)
            code_start=$(echo "$file_content" | grep -m 1 -v -E "
^
[[:space:]]*(//|/\*|#|$)")
            ;;
    esac
    
    # Write the license header
    echo "$license_header" > "$tmp_file"
    echo "" >> "$tmp_file"  # Add blank line after license
    
    # Write the code content
    if [ -n "$code_start" ]; then
        # Need to get the line number to extract from that point
        local start_line=$(echo "$file_content" | grep -n -m 1 -F "$code_start" | cut -d: -f1)
        if [ -n "$start_line" ]; then
            echo "$file_content" | tail -n +$start_line >> "$tmp_file"
        else
            # Fallback if we can't find the line number
            echo "$file_content" >> "$tmp_file"
        fi
    else
        # No code markers found, just append the whole content
        echo "$file_content" >> "$tmp_file"
    fi
    
    # Replace the file and stage changes
    cp "$tmp_file" "$file"
    rm "$tmp_file"
    git add "$file"
}

# Fix specific problematic files that still have duplicates
clean_file "iOS/Views/Home/HomeViewControllerExtensions.swift" "swift"
clean_file "Shared/Magic/zsign/common/base64.cpp" "cpp"
clean_file "Shared/Magic/zsign/common/base64.h" "cpp"

# Find any other files that might have duplicate headers
echo "Scanning for files with multiple license headers..."
files_with_duplicates=$(grep -l -A 1 -B 1 "Proprietary Software License" --include="*.swift" --include="*.h" --include="*.mm" --include="*.cpp" --include="*.hpp" -r . | xargs -I{} grep -c "Proprietary Software License" {} 2>/dev/null | grep -v ":1$")

if [ -n "$files_with_duplicates" ]; then
    echo "Found files with duplicate license headers:"
    echo "$files_with_duplicates"
    
    for file_count in $files_with_duplicates; do
        IFS=':' read -r file count <<< "$file_count"
        if [[ "$file" == *.swift ]]; then
            clean_file "$file" "swift"
        elif [[ "$file" == *.cpp || "$file" == *.c ]]; then
            clean_file "$file" "cpp"
        elif [[ "$file" == *.h ]]; then
            clean_file "$file" "cpp"
        elif [[ "$file" == *.mm ]]; then
            clean_file "$file" "cpp"
        elif [[ "$file" == *.hpp ]]; then
            clean_file "$file" "cpp"
        elif [[ "$file" == *.sh ]]; then
            clean_file "$file" "shell"
        else
            clean_file "$file" "swift"  # Default to Swift header
        fi
    done
fi

# Commit changes if any
if git diff --cached --quiet; then
    echo "No changes to commit"
else
    git commit -m "Final cleanup of duplicate license headers"
    echo "Changes committed."
fi
