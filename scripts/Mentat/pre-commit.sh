#!/bin/bash
set -euo pipefail

echo "Running pre-commit checks on $(date)..."

# Environment variables
readonly IS_CI=${CI:-false}
readonly TIMEOUT=${TIMEOUT:-180}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run commands with timeout
run_with_timeout() {
    local timeout="$1"
    local cmd="$2"
    echo "Executing: $cmd"
    if command_exists timeout; then
        timeout "$timeout" bash -c "$cmd" || { echo "Warning: Command timed out or failed: $cmd" >&2; return 1; }
    else
        bash -c "$cmd" || { echo "Warning: Command failed: $cmd" >&2; return 1; }
    fi
}

# Format Swift code
if command_exists swiftformat; then
    echo "Formatting Swift files with SwiftFormat (version: $(swiftformat --version))..."
    run_with_timeout "$TIMEOUT" "swiftformat . --exclude Pods --exclude .build --exclude .swiftpm --exclude Carthage --exclude vendor"
else
    echo "Warning: SwiftFormat not found. Skipping Swift formatting." >&2
fi

# Lint Swift code
if command_exists swiftlint; then
    echo "Linting Swift files with SwiftLint (version: $(swiftlint version))..."
    [[ -f .swiftlint.yml ]] && run_with_timeout "$TIMEOUT" "swiftlint --fix" || echo "Warning: No SwiftLint configuration found." >&2
else
    echo "Warning: SwiftLint not found. Skipping Swift linting." >&2
fi

# Format C/C++/Objective-C files
if command_exists clang-format; then
    echo "Formatting C/C++/Objective-C files with clang-format (version: $(clang-format --version))..."
    find . -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.m" -o -name "*.mm" \) \
        -not -path "*/Pods/*" \
        -not -path "*/.build/*" \
        -not -path "*/Carthage/*" \
        -not -path "*/vendor/*" \
        -print0 | while IFS= read -r -d '' file; do
            echo "Formatting $file"
            clang-format -i "$file" || echo "Warning: Failed to format $file" >&2
        done
else
    echo "Warning: clang-format not found. Skipping C/C++/Objective-C formatting." >&2
fi

# Build check (skipped in CI)
if [[ "$IS_CI" != "true" && $(command_exists xcodebuild) ]]; then
    echo "Performing build check..."
    if [[ -d "backdoor.xcworkspace" ]]; then
        run_with_timeout "$TIMEOUT" "xcodebuild -workspace backdoor.xcworkspace -scheme 'backdoor (Debug)' -destination 'platform=iOS Simulator,name=iPhone 14' clean build CODE_SIGNING_ALLOWED=NO"
    elif [[ -d "backdoor.xcodeproj" ]]; then
        run_with_timeout "$TIMEOUT" "xcodebuild -project backdoor.xcodeproj -scheme 'backdoor (Debug)' -destination 'platform=iOS Simulator,name=iPhone 14' clean build CODE_SIGNING_ALLOWED=NO"
    else
        echo "Warning: No Xcode project/workspace found. Skipping build check." >&2
    fi
else
    echo "Skipping build check in CI environment or xcodebuild not available."
fi

echo "Pre-commit checks completed successfully on $(date)!"
exit 0
