#!/bin/bash

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format is not installed"
    echo "Please install clang-format to use this pre-commit hook"
    exit 1
fi

# Copy the pre-commit hook
cp .githooks/pre-commit .git/hooks/

# Make the hook executable
chmod +x .git/hooks/pre-commit

echo "Pre-commit hook installed successfully!"
echo "The hook will now run clang-format on all staged .cpp and .hpp files before each commit." 