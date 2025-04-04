# Development Tools

This directory contains scripts related to development tooling and environment setup.

## Available Scripts

### dev-tools.sh

A comprehensive script for setting up and maintaining the development environment.

#### Features:
- Installs and configures development tools (SwiftLint, SwiftFormat, clang-format)
- Sets up configuration files (.swiftlint.yml, .swiftformat, .clang-format)
- Formats Swift, Objective-C, and C++ code
- Lints Swift code
- Updates .gitignore with development-specific entries

#### Usage:
```bash
./scripts/dev/dev-tools.sh [command]
```

#### Commands:
- `setup`: Install development tools and configurations
- `format`: Format Swift and C++/Objective-C code
- `lint`: Lint Swift code and show issues
- `check`: Run all code quality checks (format + lint)
- `help`: Show help message
