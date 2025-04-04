#!/bin/bash

# Script to fix license headers in all code files
# This script addresses issues with duplicate license headers and removes them

# Define the license text for different file types
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

SHELL_HEADER="#!/bin/bash
#
# Proprietary Software License Version 1.0
#
# Copyright (C) 2025 BDG
#
# Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
#"

# Create a temporary file to track processed files
PROCESSED_FILES_LIST=$(mktemp)
echo "0" > "$PROCESSED_FILES_LIST"

# Function to fix a file's license header
fix_license_header() {
    local file=$1
    local header=$2
    local keep_shebang=$3
    local tmp_file=$(mktemp)
    local is_modified=false
    local file_content
    
    echo "Processing $file..."
    
    # Skip files containing "LICENSE" in the filename (for safety)
    if [[ "$(basename "$file")" == *LICENSE* ]]; then
        echo "  Skipping license file: $file"
        return
    fi
    
    # Skip this script itself
    if [[ "$(basename "$file")" == "fix_license_headers.sh" ]]; then
        echo "  Skipping this script itself"
        return
    fi
    
    # Determine if it's a text file
    if ! file "$file" | grep -q "text"; then
        echo "  Skipping non-text file: $file"
        return
    fi
    
    # Read file content, strip any existing duplicate license headers
    file_content=$(cat "$file")
    
    # Count occurrences of license header to detect duplicates
    license_count=$(grep -c "Proprietary Software License" "$file")
    
    if [ "$license_count" -gt 1 ]; then
        echo "  Found $license_count duplicate license headers, fixing..."
        is_modified=true
        
        # Backup original file first
        cp "$file" "${file}.bak"
        
        # Remove all license headers and empty lines at the top
        file_content=$(echo "$file_content" | sed '/Proprietary Software License/,/License\./d')
        
        # Detect if file has shebang and needs to preserve it
        if [[ "$keep_shebang" == "true" ]] && echo "$file_content" | head -n 1 | grep -q "^#!"; then
            local shebang=$(echo "$file_content" | head -n 1)
            echo "$shebang" > "$tmp_file"
            echo "" >> "$tmp_file"
            echo "$header" >> "$tmp_file"
            echo "" >> "$tmp_file"
            
            # Add the rest of the file content after the shebang
            echo "$file_content" | tail -n +2 >> "$tmp_file"
        else
            # Add the license header at the top
            echo "$header" > "$tmp_file"
            echo "" >> "$tmp_file"
            
            # Add the file content after
            echo "$file_content" >> "$tmp_file"
        fi
    elif [ "$license_count" -eq 0 ]; then
        echo "  No license header found, adding one..."
        is_modified=true
        
        # Detect if file has shebang and needs to preserve it
        if [[ "$keep_shebang" == "true" ]] && head -n 1 "$file" | grep -q "^#!"; then
            local shebang=$(head -n 1 "$file")
            echo "$shebang" > "$tmp_file"
            echo "" >> "$tmp_file"
            echo "$header" >> "$tmp_file"
            echo "" >> "$tmp_file"
            
            # Add the rest of the file content after the shebang
            tail -n +2 "$file" >> "$tmp_file"
        else
            # Add the license header at the top
            echo "$header" > "$tmp_file"
            echo "" >> "$tmp_file"
            
            # Add the file content
            cat "$file" >> "$tmp_file"
        fi
    else
        echo "  License header is fine, no changes needed."
        rm "$tmp_file"
        return
    fi
    
    # Only replace the file if we actually made changes
    if [ "$is_modified" = true ]; then
        mv "$tmp_file" "$file"
        
        # Check if the file still has content after our changes
        if [ ! -s "$file" ]; then
            echo "  WARNING: File is empty after processing, restoring from backup..."
            if [ -f "${file}.bak" ]; then
                cp "${file}.bak" "$file"
            fi
        fi
        
        # Increment the processed files counter in our tracking file
        CURRENT_COUNT=$(cat "$PROCESSED_FILES_LIST")
        echo $((CURRENT_COUNT + 1)) > "$PROCESSED_FILES_LIST"
        echo "  Updated license header in $file"
    else
        rm "$tmp_file"
    fi
    
    # Remove backup if it exists
    if [ -f "${file}.bak" ]; then
        rm "${file}.bak"
    fi
}

# Function to specifically fix HomeViewControllerExtensions.swift if it was corrupted
fix_home_extensions() {
    echo "Checking if HomeViewControllerExtensions.swift needs restoration..."
    
    # Check if file exists and has only license headers
    if [ -f "iOS/Views/Home/HomeViewControllerExtensions.swift" ]; then
        local code_count=$(grep -v "Proprietary Software License\|Copyright\|^//\|^$" "iOS/Views/Home/HomeViewControllerExtensions.swift" | wc -l)
        
        if [ "$code_count" -lt 5 ]; then
            echo "HomeViewControllerExtensions.swift appears corrupted, fixing..."
            
            # Create proper content - add missing implementations
            cat > "iOS/Views/Home/HomeViewControllerExtensions.swift" << 'EOF'
//
// Proprietary Software License Version 1.0
//
// Copyright (C) 2025 BDG
//
// Backdoor App Signer is proprietary software. You may not use, modify, or distribute it except as expressly permitted under the terms of the Proprietary Software License.
//

import UIKit

// Extension to add protocol conformance to HomeViewController
extension HomeViewController {
    
    // MARK: - UITableViewDragDelegate
    
    func tableView(_ tableView: UITableView, itemsForBeginning session: UIDragSession, at indexPath: IndexPath) -> [UIDragItem] {
        let file = searchController.isActive ? filteredFileList[indexPath.row] : fileList[indexPath.row]
        let itemProvider = NSItemProvider(object: file.url as NSURL)
        let dragItem = UIDragItem(itemProvider: itemProvider)
        dragItem.localObject = file
        return [dragItem]
    }
    
    // MARK: - UITableViewDropDelegate
    
    func tableView(_ tableView: UITableView, performDropWith coordinator: UITableViewDropCoordinator) {
        // Implementation for drop handling
        coordinator.session.loadObjects(ofClass: NSURL.self) { items in
            guard let urls = items as? [URL] else { return }
            
            for url in urls {
                self.handleImportedFile(url: url)
            }
        }
    }
    
    func tableView(_ tableView: UITableView, dropSessionDidUpdate session: UIDropSession, withDestinationIndexPath destinationIndexPath: IndexPath?) -> UITableViewDropProposal {
        return UITableViewDropProposal(operation: .copy, intent: .insertAtDestinationIndexPath)
    }
}
EOF
            echo "Fixed HomeViewControllerExtensions.swift with proper code"
            CURRENT_COUNT=$(cat "$PROCESSED_FILES_LIST")
            echo $((CURRENT_COUNT + 1)) > "$PROCESSED_FILES_LIST"
        fi
    fi
}

# Function to process a directory
process_directory() {
    local dir=$1
    echo "Scanning directory: $dir"
    
    # Process Swift files
    find "$dir" -type f -name "*.swift" | while read file; do
        fix_license_header "$file" "$SWIFT_HEADER" "false"
    done
    
    # Process Objective-C header files
    find "$dir" -type f -name "*.h" | while read file; do
        fix_license_header "$file" "$CPP_HEADER" "false"
    done
    
    # Process Objective-C++ files
    find "$dir" -type f -name "*.mm" | while read file; do
        fix_license_header "$file" "$CPP_HEADER" "false"
    done
    
    # Process C++ files
    find "$dir" -type f -name "*.cpp" | while read file; do
        fix_license_header "$file" "$CPP_HEADER" "false"
    done
    
    # Process C++ header files
    find "$dir" -type f -name "*.hpp" | while read file; do
        fix_license_header "$file" "$CPP_HEADER" "false"
    done
    
    # Process Shell scripts (preserving shebang)
    find "$dir" -type f -name "*.sh" | grep -v "fix_license_headers.sh" | while read file; do
        fix_license_header "$file" "$SHELL_HEADER" "true"
    done
}

# Main execution
echo "Starting license header fix process..."

# Fix known problematic files first
fix_home_extensions

# Process main directories
process_directory "iOS"
process_directory "Shared"

# Process root shell scripts
find . -maxdepth 1 -type f -name "*.sh" | grep -v "fix_license_headers.sh" | while read file; do
    fix_license_header "$file" "$SHELL_HEADER" "true"
done

# Get the final count from our tracking file
FINAL_COUNT=$(cat "$PROCESSED_FILES_LIST")
echo "Completed! $FINAL_COUNT files have been fixed with proper license headers."

# Clean up temp file
rm "$PROCESSED_FILES_LIST"
