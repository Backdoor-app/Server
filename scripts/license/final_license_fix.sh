#!/bin/bash

# Final script to fix any remaining license header issues

# Function to clean a file completely
clean_file() {
    local file=$1
    local file_type=$2
    local tmp_file=$(mktemp)
    
    echo "Cleaning $file..."
    
    # Check for merge conflict markers
    if grep -q "<<<<<<< HEAD" "$file"; then
        echo "  Found merge conflict markers, resolving..."
    fi
    
    # Select the appropriate license header based on file type
    if [[ "$file_type" == "swift" ]]; then
        license_header="//
// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
//"
    elif [[ "$file_type" == "cpp" || "$file_type" == "h" || "$file_type" == "mm" || "$file_type" == "hpp" ]]; then
        license_header="/*
 * Proprietary Software License Version 1.0
 *
 * Copyright (C) 2025 BDG
 *
 * Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
 */"
    else
        license_header="# Proprietary Software License Version 1.0
#
# Copyright (C) 2025 BDG
#
# Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License."
    fi
    
    # Find where the actual code starts by looking for imports, class declarations, etc.
    local start_line
    
    # Skip past any license headers, comments, merge markers, etc.
    if [[ "$file_type" == "swift" ]]; then
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
let" "$file" | cut -d: -f1)
    elif [[ "$file_type" == "cpp" || "$file_type" == "h" || "$file_type" == "mm" || "$file_type" == "hpp" ]]; then
        start_line=$(grep -n -m 1 -E "
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
typedef" "$file" | cut -d: -f1)
    else
        # For other file types, find any line that's not a comment, empty, or merge marker
        start_line=$(grep -n -m 1 -v -E "
^
[[:space:]]*(//|/\*|#|$|<<<<<<|=======|>>>>>>>)" "$file" | cut -d: -f1)
    fi
    
    # Add the clean license header
    echo "$license_header" > "$tmp_file"
    echo "" >> "$tmp_file"  # Add a blank line after license
    
    # Add the code content if found
    if [ -n "$start_line" ]; then
        # Extract the code starting from the identified line
        tail -n +$start_line "$file" >> "$tmp_file"
    else
        # If no code start marker found, look for a line without comment syntax or merge markers
        grep -v -E "(//|/\*|\*/|#|<<<<<<|=======|>>>>>>>)" "$file" > "$tmp_file"
    fi
    
    # Replace the original file
    cp "$tmp_file" "$file"
    rm "$tmp_file"
    git add "$file"
    return 0
}

# Fix specific problem files we identified
clean_file "iOS/Views/Home/HomeViewControllerExtensions.swift" "swift"
clean_file "Shared/Magic/zsign/common/base64.cpp" "cpp"
clean_file "Shared/Magic/zsign/common/base64.h" "h"

# Find any files with merge conflict markers
echo "Checking for files with merge conflict markers..."
conflict_files=$(grep -l -E "<<<<<<< HEAD|>>>>>>> " --include="*.swift" --include="*.h" --include="*.mm" --include="*.cpp" --include="*.hpp" -r .)

if [ -n "$conflict_files" ]; then
    echo "Found files with merge conflicts:"
    echo "$conflict_files"
    echo ""
    
    echo "$conflict_files" | while read file; do
        if [[ "$file" == *.swift ]]; then
            clean_file "$file" "swift"
        elif [[ "$file" == *.cpp || "$file" == *.c ]]; then
            clean_file "$file" "cpp"
        elif [[ "$file" == *.h ]]; then
            clean_file "$file" "h"
        elif [[ "$file" == *.mm ]]; then
            clean_file "$file" "mm"
        elif [[ "$file" == *.hpp ]]; then
            clean_file "$file" "hpp"
        else
            clean_file "$file" "other"
        fi
    done
fi

# Commit changes if any files were fixed
if git diff --cached --quiet; then
    echo "No changes to commit"
else
    git commit -m "Final fix for license headers and merge conflicts"
    echo "Changes committed."
fi
