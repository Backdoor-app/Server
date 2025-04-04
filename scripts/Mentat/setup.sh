#!/bin/bash
set -euo pipefail  # Enable strict mode: exit on error, undefined vars, and pipe failures

echo "Starting setup process on $(date)..."

# Environment variables
export PATH="$HOME/.local/bin:$PATH"
readonly TIMEOUT=${TIMEOUT:-180}  # Default timeout in seconds

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install SwiftLint
install_swiftlint() {
    echo "Installing SwiftLint..."
    if command_exists swiftlint; then
        echo "SwiftLint already installed (version: $(swiftlint version))"
        return 0
    fi

    local url=$(curl -s https://api.github.com/repos/realm/SwiftLint/releases/latest \
        | grep browser_download_url \
        | grep portable \
        | cut -d '"' -f 4)
    
    if [[ -z "$url" ]]; then
        echo "Error: Failed to fetch SwiftLint download URL" >&2
        return 1
    fi

    curl -L "$url" -o swiftlint.zip || { echo "Error: Download failed" >&2; return 1; }
    unzip -o swiftlint.zip -d swiftlint_temp || { echo "Error: Unzip failed" >&2; return 1; }
    chmod +x swiftlint_temp/swiftlint
    mkdir -p "$HOME/.local/bin"
    mv swiftlint_temp/swiftlint "$HOME/.local/bin/"
    rm -rf swiftlint_temp swiftlint.zip
    echo "SwiftLint installed successfully (version: $(swiftlint version))"
}

# Function to install SwiftFormat
install_swiftformat() {
    echo "Installing SwiftFormat..."
    if command_exists swiftformat; then
        echo "SwiftFormat already installed (version: $(swiftformat --version))"
        return 0
    fi

    local url=$(curl -s https://api.github.com/repos/nicklockwood/SwiftFormat/releases/latest \
        | grep browser_download_url \
        | grep -v artifactbundle \
        | grep -E "swiftformat$" \
        | head -n 1 \
        | cut -d '"' -f 4)
    
    if [[ -z "$url" ]]; then
        echo "Error: Failed to fetch SwiftFormat download URL" >&2
        return 1
    fi

    curl -L "$url" -o "$HOME/.local/bin/swiftformat" || { echo "Error: Download failed" >&2; return 1; }
    chmod +x "$HOME/.local/bin/swiftformat"
    echo "SwiftFormat installed successfully (version: $(swiftformat --version))"
}

# Function to install clang-format
install_clang_format() {
    echo "Installing clang-format..."
    if command_exists clang-format; then
        echo "clang-format already installed (version: $(clang-format --version))"
        return 0
    fi

    if command_exists apt-get; then
        apt-get update -y && apt-get install -y clang-format
    elif command_exists yum; then
        yum install -y clang-tools-extra
    elif command_exists brew; then
        brew install clang-format
    else
        echo "Warning: Could not install clang-format automatically. Please install manually." >&2
        return 1
    fi
    echo "clang-format installed successfully (version: $(clang-format --version))"
}

# Function to create configuration file if it doesn't exist
create_config() {
    local file="$1"
    local content="$2"
    if [[ ! -f "$file" ]]; then
        echo "Creating $file..."
        echo "$content" > "$file"
    else
        echo "$file already exists, skipping creation"
    fi
}

# Main installation process
install_swiftlint || echo "SwiftLint installation failed" >&2
install_swiftformat || echo "SwiftFormat installation failed" >&2
install_clang_format || echo "clang-format installation failed" >&2

# Create configuration files
create_config ".swiftlint.yml" "$(cat << 'EOF'
disabled_rules:
  - trailing_whitespace
  - line_length
  - cyclomatic_complexity
  - function_body_length
  - file_length
  - force_cast
  - type_body_length
included:
  - iOS
  - Shared
excluded:
  - Pods
  - .build
  - .swiftpm
  - Carthage
  - vendor
opt_in_rules:
  - empty_count
  - empty_string
  - closure_spacing
reporter: "xcode"
EOF
)"

create_config ".swiftformat" "$(cat << 'EOF'
--indent 4
--indentcase true
--trimwhitespace always
--importgrouping alphabetized
--semicolons never
--header strip
--disable redundantSelf
--swiftversion 5.9
--wraparguments beforefirst
--wrapparameters beforefirst
EOF
)"

create_config ".clang-format" "$(cat << 'EOF'
BasedOnStyle: LLVM
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 120
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
IndentCaseLabels: true
AccessModifierOffset: -4
PointerAlignment: Left
NamespaceIndentation: All
BreakBeforeBraces: Allman
EOF
)"

# Resolve dependencies
if [[ -f "Package.swift" && $(command_exists swift) ]]; then
    echo "Resolving Swift Package Manager dependencies..."
    swift package resolve || echo "Warning: Failed to resolve Swift packages" >&2
elif [[ -d "backdoor.xcworkspace" && $(command_exists xcodebuild) ]]; then
    echo "Resolving Xcode workspace dependencies..."
    xcodebuild -resolvePackageDependencies -workspace backdoor.xcworkspace || echo "Warning: Failed to resolve Xcode dependencies" >&2
fi

echo "Setup completed successfully on $(date)!"
exit 0
