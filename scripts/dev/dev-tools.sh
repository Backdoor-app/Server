#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  Backdoor Development Tools Script    ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Function to print usage instructions
print_usage() {
    echo -e "\n${GREEN}Usage:${NC}"
    echo -e "  ./dev-tools.sh [command]"
    echo -e "\n${GREEN}Commands:${NC}"
    echo -e "  ${YELLOW}setup${NC}      - Install development tools and configurations"
    echo -e "  ${YELLOW}format${NC}     - Format Swift and C++/Objective-C code"
    echo -e "  ${YELLOW}lint${NC}       - Lint Swift code and show issues"
    echo -e "  ${YELLOW}check${NC}      - Run all code quality checks (format + lint)"
    echo -e "  ${YELLOW}help${NC}       - Show this help message"
    echo -e "\n${GREEN}Examples:${NC}"
    echo -e "  ./dev-tools.sh setup    # Install all development tools"
    echo -e "  ./dev-tools.sh format   # Format all code files"
    echo -e "  ./dev-tools.sh check    # Run all code quality checks"
    echo -e "\n${BLUE}Note:${NC} Run setup once when starting development or setting up a new environment."
}

# Function to install SwiftLint
install_swiftlint() {
    echo -e "\n${BLUE}Installing SwiftLint...${NC}"
    
    # Check if SwiftLint is already available
    if command -v swiftlint &> /dev/null; then
        echo -e "${GREEN}SwiftLint already installed${NC}"
        return
    fi
    
    # Try to download pre-built binary
    LATEST_SWIFTLINT_URL=$(curl -s https://api.github.com/repos/realm/SwiftLint/releases/latest | grep browser_download_url | grep portable | cut -d '"' -f 4)
    
    if [ ! -z "$LATEST_SWIFTLINT_URL" ]; then
        echo "Downloading SwiftLint from $LATEST_SWIFTLINT_URL"
        curl -L "$LATEST_SWIFTLINT_URL" -o swiftlint.zip
        unzip -o swiftlint.zip
        if [ -f "swiftlint" ]; then
            chmod +x swiftlint
            mkdir -p $HOME/.local/bin
            mv swiftlint $HOME/.local/bin/
            rm -f swiftlint.zip LICENSE 2>/dev/null || true
            export PATH="$HOME/.local/bin:$PATH"
            echo -e "${GREEN}SwiftLint installed successfully${NC}"
        else
            echo -e "${RED}Error: SwiftLint binary not found in the downloaded package${NC}"
        fi
    else
        echo -e "${RED}Error: Could not find SwiftLint download URL${NC}"
    fi
}

# Function to install SwiftFormat
install_swiftformat() {
    echo -e "\n${BLUE}Installing SwiftFormat...${NC}"
    
    # Check if SwiftFormat is already available
    if command -v swiftformat &> /dev/null; then
        echo -e "${GREEN}SwiftFormat already installed${NC}"
        return
    fi
    
    # Try direct binary download first
    LATEST_SWIFTFORMAT_URL=$(curl -s https://api.github.com/repos/nicklockwood/SwiftFormat/releases/latest | \
        grep browser_download_url | \
        grep -v artifactbundle | \
        grep -v .zip | \
        grep -E "swiftformat$" | \
        head -n 1 | \
        cut -d '"' -f 4)
    
    if [ ! -z "$LATEST_SWIFTFORMAT_URL" ]; then
        echo "Found direct SwiftFormat binary at $LATEST_SWIFTFORMAT_URL"
        curl -L "$LATEST_SWIFTFORMAT_URL" -o swiftformat
        chmod +x swiftformat
        mkdir -p $HOME/.local/bin
        mv swiftformat $HOME/.local/bin/
        export PATH="$HOME/.local/bin:$PATH"
        echo -e "${GREEN}SwiftFormat installed successfully${NC}"
        return
    fi
    
    # Try artifact bundle as fallback
    echo "No direct binary found, trying artifact bundle..."
    BUNDLE_URL=$(curl -s https://api.github.com/repos/nicklockwood/SwiftFormat/releases/latest | \
        grep browser_download_url | \
        grep artifactbundle | \
        head -n 1 | \
        cut -d '"' -f 4)
    
    if [ ! -z "$BUNDLE_URL" ]; then
        echo "Found SwiftFormat artifact bundle at $BUNDLE_URL"
        TEMP_DIR=$(mktemp -d)
        curl -L "$BUNDLE_URL" -o "$TEMP_DIR/swiftformat.zip"
        unzip -o "$TEMP_DIR/swiftformat.zip" -d "$TEMP_DIR"
        
        # Try to find the correct binary for this platform
        if [[ "$(uname)" == "Darwin" ]]; then
            # macOS
            SWIFTFORMAT_BIN=$(find "$TEMP_DIR" -name "swiftformat" -type f | grep -v linux | head -n 1)
        else
            # Linux - try to match architecture
            if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
                SWIFTFORMAT_BIN=$(find "$TEMP_DIR" -name "*aarch64*" -type f | head -n 1)
            else
                SWIFTFORMAT_BIN=$(find "$TEMP_DIR" -name "*linux*" -type f | grep -v aarch64 | head -n 1)
            fi
        fi
        
        if [ ! -z "$SWIFTFORMAT_BIN" ]; then
            echo "Found binary at $SWIFTFORMAT_BIN"
            chmod +x "$SWIFTFORMAT_BIN"
            mkdir -p $HOME/.local/bin
            cp "$SWIFTFORMAT_BIN" "$HOME/.local/bin/swiftformat"
            export PATH="$HOME/.local/bin:$PATH"
            echo -e "${GREEN}SwiftFormat installed successfully${NC}"
        else
            echo -e "${RED}Error: Could not find SwiftFormat binary in artifact bundle${NC}"
        fi
        
        rm -rf "$TEMP_DIR"
    else
        echo -e "${RED}Error: Could not find SwiftFormat download URL${NC}"
    fi
}

# Function to install clang-format
install_clang_format() {
    echo -e "\n${BLUE}Installing clang-format...${NC}"
    
    # Check if clang-format is already available
    if command -v clang-format &> /dev/null; then
        echo -e "${GREEN}clang-format already installed${NC}"
        return
    fi
    
    # Try apt-get for Debian-based systems
    if command -v apt-get &> /dev/null; then
        echo "Trying to install clang-format via apt-get..."
        sudo apt-get update -y
        sudo apt-get install -y clang-format || echo -e "${RED}Failed to install clang-format via apt-get${NC}"
        return
    fi
    
    # Try brew for macOS
    if command -v brew &> /dev/null; then
        echo "Trying to install clang-format via brew..."
        brew install clang-format || echo -e "${RED}Failed to install clang-format via brew${NC}"
        return
    fi
    
    # Try yum for Red Hat-based systems
    if command -v yum &> /dev/null; then
        echo "Trying to install clang-format via yum..."
        sudo yum install -y clang-tools-extra || echo -e "${RED}Failed to install clang-format via yum${NC}"
        return
    fi
    
    echo -e "${YELLOW}Could not install clang-format automatically. Please install it manually.${NC}"
}

# Setup configuration files
setup_configurations() {
    echo -e "\n${BLUE}Setting up configuration files...${NC}"
    
    # Initialize SwiftLint configuration if not exists
    if ! [ -f .swiftlint.yml ]; then
        echo "Creating default SwiftLint configuration..."
        cat > .swiftlint.yml << 'SWIFTLINT_CONFIG'
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
SWIFTLINT_CONFIG
        echo -e "${GREEN}Created .swiftlint.yml${NC}"
    fi

    # Initialize SwiftFormat configuration if not exists
    if ! [ -f .swiftformat ]; then
        echo "Creating default SwiftFormat configuration..."
        cat > .swiftformat << 'SWIFTFORMAT_CONFIG'
--indent 4
--indentcase true
--trimwhitespace always
--importgrouping alphabetized
--semicolons never
--header strip
--disable redundantSelf
SWIFTFORMAT_CONFIG
        echo -e "${GREEN}Created .swiftformat${NC}"
    fi

    # Initialize .clang-format for C++/Objective-C files
    if ! [ -f .clang-format ]; then
        echo "Creating default .clang-format configuration..."
        cat > .clang-format << 'CLANG_FORMAT_CONFIG'
BasedOnStyle: LLVM
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 120
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
IndentCaseLabels: true
AccessModifierOffset: -4
CLANG_FORMAT_CONFIG
        echo -e "${GREEN}Created .clang-format${NC}"
    fi
}

# Update gitignore
update_gitignore() {
    echo -e "\n${BLUE}Updating .gitignore...${NC}"
    
    # Check if the entries already exist
    if grep -q "# Development tools" .gitignore; then
        echo -e "${GREEN}Development tools already in .gitignore${NC}"
        return
    fi
    
    # Add development tool entries to .gitignore
    cat >> .gitignore << 'GITIGNORE_CONTENT'

# Development tools
.swiftpm/
.build/
*.swp
*~
.swiftlint.txt
*.swift.orig
GITIGNORE_CONTENT
    echo -e "${GREEN}Updated .gitignore${NC}"
}

# Ensure PATH includes local bin
setup_path() {
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
        echo -e "\n${BLUE}Added $HOME/.local/bin to PATH${NC}"
    fi
}

# Format Swift code
format_swift() {
    echo -e "\n${BLUE}Formatting Swift files...${NC}"
    
    if ! command -v swiftformat &> /dev/null; then
        echo -e "${RED}SwiftFormat not found. Run './dev-tools.sh setup' first.${NC}"
        return 1
    fi
    
    SWIFT_FILES=$(find . -name "*.swift" -not -path "*/Pods/*" -not -path "*/.build/*" -not -path "*/.swiftpm/*")
    if [ ! -z "$SWIFT_FILES" ]; then
        swiftformat . --exclude Pods,.build,.swiftpm
        echo -e "${GREEN}Swift formatting completed${NC}"
    else
        echo -e "${YELLOW}No Swift files found to format.${NC}"
    fi
}

# Format C++/Objective-C code
format_cpp() {
    echo -e "\n${BLUE}Formatting C++/Objective-C/Objective-C++ files...${NC}"
    
    if ! command -v clang-format &> /dev/null; then
        echo -e "${RED}clang-format not found. Run './dev-tools.sh setup' first.${NC}"
        return 1
    fi
    
    CPP_FILES=$(find . -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.m" -o -name "*.mm" \) -not -path "*/Pods/*" -not -path "*/.build/*")
    if [ ! -z "$CPP_FILES" ]; then
        for file in $CPP_FILES; do
            echo "Formatting $file"
            clang-format -i "$file" || echo -e "${RED}Failed to format $file${NC}"
        done
        echo -e "${GREEN}C++/Objective-C formatting completed${NC}"
    else
        echo -e "${YELLOW}No C++/Objective-C/Objective-C++ files found to format.${NC}"
    fi
}

# Lint Swift code
lint_swift() {
    echo -e "\n${BLUE}Linting Swift files...${NC}"
    
    if ! command -v swiftlint &> /dev/null; then
        echo -e "${RED}SwiftLint not found. Run './dev-tools.sh setup' first.${NC}"
        return 1
    fi
    
    SWIFT_FILES=$(find . -name "*.swift" -not -path "*/Pods/*" -not -path "*/.build/*" -not -path "*/.swiftpm/*")
    if [ ! -z "$SWIFT_FILES" ]; then
        if [ -f .swiftlint.yml ]; then
            swiftlint --fix || true
            echo -e "${GREEN}Swift linting completed${NC}"
        else
            echo -e "${YELLOW}No SwiftLint configuration found, using default.${NC}"
            swiftlint --fix || true
            echo -e "${GREEN}Swift linting completed${NC}"
        fi
    else
        echo -e "${YELLOW}No Swift files found to lint.${NC}"
    fi
}

# Setup command - install all tools and create configs
setup_all() {
    echo -e "\n${BLUE}=== Setting up development environment ===${NC}"
    install_swiftlint
    install_swiftformat
    install_clang_format
    setup_configurations
    setup_path
    update_gitignore
    echo -e "\n${GREEN}Setup completed! You can now use the other commands to format and lint your code.${NC}"
}

# Format all code
format_all() {
    echo -e "\n${BLUE}=== Formatting all code ===${NC}"
    setup_path
# Check if the dev-tools.sh file exists
ls -la dev-tools.sh

# Stage the file and try to commit again
git add dev-tools.sh
git status
# Create the script again and make sure it exists
cat > dev-tools.sh << 'EOF'
#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  Backdoor Development Tools Script    ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Function to print usage instructions
print_usage() {
    echo -e "\n${GREEN}Usage:${NC}"
    echo -e "  ./dev-tools.sh [command]"
    echo -e "\n${GREEN}Commands:${NC}"
    echo -e "  ${YELLOW}setup${NC}      - Install development tools and configurations"
    echo -e "  ${YELLOW}format${NC}     - Format Swift and C++/Objective-C code"
    echo -e "  ${YELLOW}lint${NC}       - Lint Swift code and show issues"
    echo -e "  ${YELLOW}check${NC}      - Run all code quality checks (format + lint)"
    echo -e "  ${YELLOW}help${NC}       - Show this help message"
    echo -e "\n${GREEN}Examples:${NC}"
    echo -e "  ./dev-tools.sh setup    # Install all development tools"
    echo -e "  ./dev-tools.sh format   # Format all code files"
    echo -e "  ./dev-tools.sh check    # Run all code quality checks"
    echo -e "\n${BLUE}Note:${NC} Run setup once when starting development or setting up a new environment."
}

# Function to install SwiftLint
install_swiftlint() {
    echo -e "\n${BLUE}Installing SwiftLint...${NC}"
    
    # Check if SwiftLint is already available
    if command -v swiftlint &> /dev/null; then
        echo -e "${GREEN}SwiftLint already installed${NC}"
        return
    fi
    
    # Try to download pre-built binary
    LATEST_SWIFTLINT_URL=$(curl -s https://api.github.com/repos/realm/SwiftLint/releases/latest | grep browser_download_url | grep portable | cut -d '"' -f 4)
    
    if [ ! -z "$LATEST_SWIFTLINT_URL" ]; then
        echo "Downloading SwiftLint from $LATEST_SWIFTLINT_URL"
        curl -L "$LATEST_SWIFTLINT_URL" -o swiftlint.zip
        unzip -o swiftlint.zip
        if [ -f "swiftlint" ]; then
            chmod +x swiftlint
            mkdir -p $HOME/.local/bin
            mv swiftlint $HOME/.local/bin/
            rm -f swiftlint.zip LICENSE 2>/dev/null || true
            export PATH="$HOME/.local/bin:$PATH"
            echo -e "${GREEN}SwiftLint installed successfully${NC}"
        else
            echo -e "${RED}Error: SwiftLint binary not found in the downloaded package${NC}"
        fi
    else
        echo -e "${RED}Error: Could not find SwiftLint download URL${NC}"
    fi
}

# Function to install SwiftFormat
install_swiftformat() {
    echo -e "\n${BLUE}Installing SwiftFormat...${NC}"
    
    # Check if SwiftFormat is already available
    if command -v swiftformat &> /dev/null; then
        echo -e "${GREEN}SwiftFormat already installed${NC}"
        return
    fi
    
    # Try direct binary download first
    LATEST_SWIFTFORMAT_URL=$(curl -s https://api.github.com/repos/nicklockwood/SwiftFormat/releases/latest | \
        grep browser_download_url | \
        grep -v artifactbundle | \
        grep -v .zip | \
        grep -E "swiftformat$" | \
        head -n 1 | \
        cut -d '"' -f 4)
    
    if [ ! -z "$LATEST_SWIFTFORMAT_URL" ]; then
        echo "Found direct SwiftFormat binary at $LATEST_SWIFTFORMAT_URL"
        curl -L "$LATEST_SWIFTFORMAT_URL" -o swiftformat
        chmod +x swiftformat
        mkdir -p $HOME/.local/bin
        mv swiftformat $HOME/.local/bin/
        export PATH="$HOME/.local/bin:$PATH"
        echo -e "${GREEN}SwiftFormat installed successfully${NC}"
        return
    fi
    
    # Try artifact bundle as fallback
    echo "No direct binary found, trying artifact bundle..."
    BUNDLE_URL=$(curl -s https://api.github.com/repos/nicklockwood/SwiftFormat/releases/latest | \
        grep browser_download_url | \
        grep artifactbundle | \
        head -n 1 | \
        cut -d '"' -f 4)
    
    if [ ! -z "$BUNDLE_URL" ]; then
        echo "Found SwiftFormat artifact bundle at $BUNDLE_URL"
        TEMP_DIR=$(mktemp -d)
        curl -L "$BUNDLE_URL" -o "$TEMP_DIR/swiftformat.zip"
        unzip -o "$TEMP_DIR/swiftformat.zip" -d "$TEMP_DIR"
        
        # Try to find the correct binary for this platform
        if [[ "$(uname)" == "Darwin" ]]; then
            # macOS
            SWIFTFORMAT_BIN=$(find "$TEMP_DIR" -name "swiftformat" -type f | grep -v linux | head -n 1)
        else
            # Linux - try to match architecture
            if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
                SWIFTFORMAT_BIN=$(find "$TEMP_DIR" -name "*aarch64*" -type f | head -n 1)
            else
                SWIFTFORMAT_BIN=$(find "$TEMP_DIR" -name "*linux*" -type f | grep -v aarch64 | head -n 1)
            fi
        fi
        
        if [ ! -z "$SWIFTFORMAT_BIN" ]; then
            echo "Found binary at $SWIFTFORMAT_BIN"
            chmod +x "$SWIFTFORMAT_BIN"
            mkdir -p $HOME/.local/bin
            cp "$SWIFTFORMAT_BIN" "$HOME/.local/bin/swiftformat"
            export PATH="$HOME/.local/bin:$PATH"
            echo -e "${GREEN}SwiftFormat installed successfully${NC}"
        else
            echo -e "${RED}Error: Could not find SwiftFormat binary in artifact bundle${NC}"
        fi
        
        rm -rf "$TEMP_DIR"
    else
        echo -e "${RED}Error: Could not find SwiftFormat download URL${NC}"
    fi
}

# Function to install clang-format
install_clang_format() {
    echo -e "\n${BLUE}Installing clang-format...${NC}"
    
    # Check if clang-format is already available
    if command -v clang-format &> /dev/null; then
        echo -e "${GREEN}clang-format already installed${NC}"
        return
    fi
    
    # Try apt-get for Debian-based systems
    if command -v apt-get &> /dev/null; then
        echo "Trying to install clang-format via apt-get..."
        sudo apt-get update -y
        sudo apt-get install -y clang-format || echo -e "${RED}Failed to install clang-format via apt-get${NC}"
        return
    fi
    
    # Try brew for macOS
    if command -v brew &> /dev/null; then
        echo "Trying to install clang-format via brew..."
        brew install clang-format || echo -e "${RED}Failed to install clang-format via brew${NC}"
        return
    fi
    
    # Try yum for Red Hat-based systems
    if command -v yum &> /dev/null; then
        echo "Trying to install clang-format via yum..."
        sudo yum install -y clang-tools-extra || echo -e "${RED}Failed to install clang-format via yum${NC}"
        return
    fi
    
    echo -e "${YELLOW}Could not install clang-format automatically. Please install it manually.${NC}"
}

# Setup configuration files
setup_configurations() {
    echo -e "\n${BLUE}Setting up configuration files...${NC}"
    
    # Initialize SwiftLint configuration if not exists
    if ! [ -f .swiftlint.yml ]; then
        echo "Creating default SwiftLint configuration..."
        cat > .swiftlint.yml << 'SWIFTLINT_CONFIG'
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
SWIFTLINT_CONFIG
        echo -e "${GREEN}Created .swiftlint.yml${NC}"
    fi

    # Initialize SwiftFormat configuration if not exists
    if ! [ -f .swiftformat ]; then
        echo "Creating default SwiftFormat configuration..."
        cat > .swiftformat << 'SWIFTFORMAT_CONFIG'
--indent 4
--indentcase true
--trimwhitespace always
--importgrouping alphabetized
--semicolons never
--header strip
--disable redundantSelf
SWIFTFORMAT_CONFIG
        echo -e "${GREEN}Created .swiftformat${NC}"
    fi

    # Initialize .clang-format for C++/Objective-C files
    if ! [ -f .clang-format ]; then
        echo "Creating default .clang-format configuration..."
        cat > .clang-format << 'CLANG_FORMAT_CONFIG'
BasedOnStyle: LLVM
IndentWidth: 4
TabWidth: 4
UseTab: Never
ColumnLimit: 120
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
IndentCaseLabels: true
AccessModifierOffset: -4
CLANG_FORMAT_CONFIG
        echo -e "${GREEN}Created .clang-format${NC}"
    fi
}

# Update gitignore
update_gitignore() {
    echo -e "\n${BLUE}Updating .gitignore...${NC}"
    
    # Check if the entries already exist
    if grep -q "# Development tools" .gitignore; then
        echo -e "${GREEN}Development tools already in .gitignore${NC}"
        return
    fi
    
    # Add development tool entries to .gitignore
    cat >> .gitignore << 'GITIGNORE_CONTENT'

# Development tools
.swiftpm/
.build/
*.swp
*~
.swiftlint.txt
*.swift.orig
GITIGNORE_CONTENT
    echo -e "${GREEN}Updated .gitignore${NC}"
}

# Ensure PATH includes local bin
setup_path() {
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
        echo -e "\n${BLUE}Added $HOME/.local/bin to PATH${NC}"
    fi
}

# Format Swift code
format_swift() {
    echo -e "\n${BLUE}Formatting Swift files...${NC}"
    
    if ! command -v swiftformat &> /dev/null; then
        echo -e "${RED}SwiftFormat not found. Run './dev-tools.sh setup' first.${NC}"
        return 1
    fi
    
    SWIFT_FILES=$(find . -name "*.swift" -not -path "*/Pods/*" -not -path "*/.build/*" -not -path "*/.swiftpm/*")
    if [ ! -z "$SWIFT_FILES" ]; then
        swiftformat . --exclude Pods,.build,.swiftpm
        echo -e "${GREEN}Swift formatting completed${NC}"
    else
        echo -e "${YELLOW}No Swift files found to format.${NC}"
    fi
}

# Format C++/Objective-C code
format_cpp() {
    echo -e "\n${BLUE}Formatting C++/Objective-C/Objective-C++ files...${NC}"
    
    if ! command -v clang-format &> /dev/null; then
        echo -e "${RED}clang-format not found. Run './dev-tools.sh setup' first.${NC}"
        return 1
    fi
    
    CPP_FILES=$(find . -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.m" -o -name "*.mm" \) -not -path "*/Pods/*" -not -path "*/.build/*")
    if [ ! -z "$CPP_FILES" ]; then
        for file in $CPP_FILES; do
            echo "Formatting $file"
            clang-format -i "$file" || echo -e "${RED}Failed to format $file${NC}"
        done
        echo -e "${GREEN}C++/Objective-C formatting completed${NC}"
    else
        echo -e "${YELLOW}No C++/Objective-C/Objective-C++ files found to format.${NC}"
    fi
}

# Lint Swift code
lint_swift() {
    echo -e "\n${BLUE}Linting Swift files...${NC}"
    
    if ! command -v swiftlint &> /dev/null; then
        echo -e "${RED}SwiftLint not found. Run './dev-tools.sh setup' first.${NC}"
        return 1
    fi
    
    SWIFT_FILES=$(find . -name "*.swift" -not -path "*/Pods/*" -not -path "*/.build/*" -not -path "*/.swiftpm/*")
    if [ ! -z "$SWIFT_FILES" ]; then
        if [ -f .swiftlint.yml ]; then
            swiftlint --fix || true
            echo -e "${GREEN}Swift linting completed${NC}"
        else
            echo -e "${YELLOW}No SwiftLint configuration found, using default.${NC}"
            swiftlint --fix || true
            echo -e "${GREEN}Swift linting completed${NC}"
        fi
    else
        echo -e "${YELLOW}No Swift files found to lint.${NC}"
    fi
}

# Setup command - install all tools and create configs
setup_all() {
    echo -e "\n${BLUE}=== Setting up development environment ===${NC}"
    install_swiftlint
    install_swiftformat
    install_clang_format
    setup_configurations
    setup_path
    update_gitignore
    echo -e "\n${GREEN}Setup completed! You can now use the other commands to format and lint your code.${NC}"
}

# Format all code
format_all() {
    echo -e "\n${BLUE}=== Formatting all code ===${NC}"
    setup_path
    format_swift
    format# Verify the dev-tools.sh file exists and is executable
ls -la dev-tools.sh

# Run the script in format and lint mode to actually fix the code
./dev-tools.sh check

# See what changes were made
git status
# Check the current directory contents
ls -la

# Try to create the dev-tools.sh script again in a different way
echo '#!/bin/bash
set -e

echo "Running code quality checks and fixes..."

# Install SwiftLint if not already available
install_swiftlint() {
  if ! command -v swiftlint &> /dev/null; then
    echo "Installing SwiftLint..."
    SWIFTLINT_URL="https://github.com/realm/SwiftLint/releases/download/0.54.0/portable_swiftlint.zip"
    curl -L "$SWIFTLINT_URL" -o swiftlint.zip
    unzip -o swiftlint.zip
    chmod +x swiftlint
    mkdir -p $HOME/.local/bin
    mv swiftlint $HOME/.local/bin/
    export PATH="$HOME/.local/bin:$PATH"
    rm -f swiftlint.zip LICENSE 2>/dev/null || true
  fi
}

# Install SwiftFormat if not already available
install_swiftformat() {
  if ! command -v swiftformat &> /dev/null; then
    echo "Installing SwiftFormat..."
    SWIFTFORMAT_URL="https://github.com/nicklockwood/SwiftFormat/releases/download/0.53.3/swiftformat.artifactbundle.zip"
    TEMP_DIR=$(mktemp -d)
    curl -L "$SWIFTFORMAT_URL" -o "$TEMP_DIR/swiftformat.zip"
    unzip -o "$TEMP_DIR/swiftformat.zip" -d "$TEMP_DIR"
    
    # Find the right binary based on platform
    if [[ "$(uname)" == "Darwin" ]]; then
      SWIFTFORMAT_BIN=$(find "$TEMP_DIR" -name "swiftformat" -type f | grep -v linux | head -n 1)
    else
      SWIFTFORMAT_BIN=$(find "$TEMP_DIR" -name "*linux*" -type f | grep -v aarch64 | head -n 1)
    fi
    
    chmod +x "$SWIFTFORMAT_BIN"
    mkdir -p $HOME/.local/bin
    cp "$SWIFTFORMAT_BIN" "$HOME/.local/bin/swiftformat"
    export PATH="$HOME/.local/bin:$PATH"
    rm -rf "$TEMP_DIR"
  fi
}

# Create SwiftLint config
create_swiftlint_config() {
  cat > .swiftlint.yml << EOF
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
    fi
}

# Setup command - install all tools and create configs
setup_all() {
    echo -e "\n${BLUE}=== Setting up development environment ===${NC}"
    install_swiftlint
    install_swiftformat
    install_clang_format
    setup_configurations
    setup_path
    update_gitignore
    echo -e "\n${GREEN}Setup completed! You can now use the other commands to format and lint your code.${NC}"
}

# Format all code
format_all() {
    echo -e "\n${BLUE}=== Formatting all code ===${NC}"
    setup_path
    format_swift
    format_cpp
    echo -e "\n${GREEN}Formatting completed!${NC}"
}

# Lint all code
lint_all() {
    echo -e "\n${BLUE}=== Linting code ===${NC}"
    setup_path
    lint_swift
    echo -e "\n${GREEN}Linting completed!${NC}"
}

# Main script execution logic
if [ $# -eq 0 ]; then
    # No arguments, show usage
    print_usage
    exit 0
fi

# Process command
command=$1

case "$command" in
    setup)
        setup_all
        ;;
    format)
        format_all
        ;;
    lint)
        lint_all
        ;;
    check)
        format_all
        lint_all
        ;;
    help)
        print_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$command'${NC}"
        print_usage
        exit 1
        ;;
esac

exit 0
