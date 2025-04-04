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


#!/bin/bash
#
#

SWIFT_HEADER="//
//"

CPP_HEADER="//
//"

# Function to add header to a file
add_header() {
    local file=$1
    local header=$2
    local tmp_file=$(mktemp)
    
    # Write the header to the temporary file
    echo "$header" > "$tmp_file"
    echo "" >> "$tmp_file"  # Add a blank line after header
    
    # Now add the file content (without any existing headers)
    # For Swift files, we'll look for the import statement as our marker
    if [[ "$file" == *.swift ]]; then
        # Find the first import statement or the first non-comment, non-empty line
        local first_line=$(grep -n -m 1 -E "
^
import|
^
[
^
//[:space:]]" "$file" | cut -d: -f1)
        
        # If no match found, assume it's line 1
        if [ -z "$first_line" ]; then
            first_line=1
        fi
        
        # Add everything from that line onwards to the temp file
        tail -n +${first_line} "$file" >> "$tmp_file"
    else
        # For other file types, look for first non-comment line
        local first_line=$(grep -n -m 1 -v "
^
[[:space:]]*//" "$file" | cut -d: -f1)
        
        # If no match found, assume it's line 1
        if [ -z "$first_line" ]; then
            first_line=1
        fi
        
        # Add everything from that line onwards to the temp file
        tail -n +${first_line} "$file" >> "$tmp_file"
    fi
    
    # Replace the original file
    mv "$tmp_file" "$file"
    echo "Added header to $file"
}

# Process Swift files
echo "Processing Swift files..."
find . -type f -name "*.swift" | while read file; do
    add_header "$file" "$SWIFT_HEADER"
done

# Process Objective-C header files
echo "Processing .h files..."
find . -type f -name "*.h" | while read file; do
    add_header "$file" "$CPP_HEADER"
done

# Process Objective-C++ files
echo "Processing .mm files..."
find . -type f -name "*.mm" | while read file; do
    add_header "$file" "$CPP_HEADER"
done

# Process C++ files
echo "Processing .cpp files..."
find . -type f -name "*.cpp" | while read file; do
    add_header "$file" "$CPP_HEADER"
done

# Process C++ header files
echo "Processing .hpp files..."
find . -type f -name "*.hpp" | while read file; do
    add_header "$file" "$CPP_HEADER"
done

echo "License headers added to all files."
