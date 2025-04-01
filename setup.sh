#!/bin/bash
set -e  # Exit on any error

# Update package lists
apt-get update

# Install dependencies for building Git
apt-get install -y build-essential libssl-dev libcurl4-gnutls-dev libexpat1-dev gettext zlib1g-dev

# Download and install the latest Git from source
GIT_VERSION="2.46.0"  # Replace with the latest version from https://github.com/git/git/releases
curl -L "https://github.com/git/git/archive/refs/tags/v${GIT_VERSION}.tar.gz" -o git.tar.gz
tar -xzf git.tar.gz
cd git-${GIT_VERSION}
make prefix=/usr/local all
make prefix=/usr/local install

# Verify installation
git --version

# Clean up
cd ..
rm -rf git-${GIT_VERSION} git.tar.gz