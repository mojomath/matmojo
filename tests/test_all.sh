#!/bin/bash
set -e  # Exit immediately if any command fails

# Find and run all test files recursively in the tests directory
find tests -name "test_*.mojo" -type f | sort | while read f; do
    echo "=========================================="
    echo "Running: $f"
    echo "=========================================="
    pixi run mojo run -I src -D ASSERT=all "$f"
done

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="