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
//
"

CPP_HEADER="//
//
"

# Function to add header to a file
add_header() {
    local file=$1
    local header=$2
    local tmp_file=$(mktemp)
    
    # Add the header
    echo -n "$header" > "$tmp_file"
    
    # Add the content (skipping any existing header comments at the top)
    # First we'll find the first line that's not a comment or blank
    grep -n -m 1 -v "
^
[[:space:]]*//" "$file" | cut -d: -f1 | xargs -I{} tail -n +{} "$file" >> "$tmp_file"
    
    # Replace the original file
    mv "$tmp_file" "$file"
    echo "Added header to $file"
}

# Process Swift files
find . -type f -name "*.swift" | while read file; do
    add_header "$file" "$SWIFT_HEADER"
done

# Process Objective-C header files
find . -type f -name "*.h" | while read file; do
    add_header "$file" "$CPP_HEADER"
done

# Process Objective-C++ files
find . -type f -name "*.mm" | while read file; do
    add_header "$file" "$CPP_HEADER"
done

# Process C++ files
find . -type f -name "*.cpp" | while read file; do
    add_header "$file" "$CPP_HEADER"
done

# Process C++ header files
find . -type f -name "*.hpp" | while read file; do
    add_header "$file" "$CPP_HEADER"
done

echo "License headers added to all files."
