# Backdoor Scripts

This directory contains various scripts used for development, maintenance, and utility tasks related to the Backdoor project.

## Directory Structure

- **dev/**: Development tools and environment setup scripts
  - `dev-tools.sh`: Comprehensive script for setting up the development environment, installing tools (SwiftLint, SwiftFormat, clang-format), and running formatting and linting tasks

- **license/**: License header management scripts
  - `add_license_header.sh`: Adds license headers to code files
  - `add_license_header_v2.sh`: Updated version of the license header addition script
  - `update_license_headers.sh`: Comprehensively updates license headers across the codebase
  - `remove_comments_add_license.sh`: Removes existing comments and adds proper license headers
  - `fix_duplicate_licenses.sh`: Fixes files with duplicate license headers
  - `final_cleanup.sh`: Performs a final cleanup of license headers
  - `final_license_fix.sh`: Fixes any remaining license header issues
  - `fix_license_headers.sh`: Main script to fix license headers in all code files

- **utils/**: Utility scripts for various maintenance tasks
  - `fix_merge_conflicts.sh`: Handles git merge conflicts while preserving license headers

## Usage

Most scripts can be run from the repository root, for example:

```bash
# Development tools
./scripts/dev/dev-tools.sh setup   # Set up development environment
./scripts/dev/dev-tools.sh format  # Format code
./scripts/dev/dev-tools.sh lint    # Lint code

# License management
./scripts/license/fix_license_headers.sh  # Fix license headers

# Utils
./scripts/utils/fix_merge_conflicts.sh  # Fix merge conflicts
```

Refer to individual scripts for more specific usage instructions.
