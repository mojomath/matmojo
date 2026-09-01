#!/bin/bash
set -e  # Exit immediately if any command fails

# Linamo does not compile without decimo --- the matrix types name
# `decimo.Numeric` --- so `tools/ensure_decimo.sh` has to have run. It normally
# finds decimo in the pixi environment and leaves `temp/` empty; `temp/` holds a
# precompiled `decimo.mojoc` only when building against a local or pinned
# checkout. Either way the directory's absence means the task never ran.
if [[ ! -d temp ]]; then
    echo "decimo has not been resolved. Run 'pixi run decimo' first." >&2
    exit 1
fi
INCLUDES=(-I src -I temp)

# Find and run all test files recursively in the tests directory
find tests -name "test_*.mojo" -type f | sort | while read f; do
    echo "=========================================="
    echo "Running: $f"
    echo "=========================================="
    pixi run mojo run "${INCLUDES[@]}" -D ASSERT=all "$f"
done

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="