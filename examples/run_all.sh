#!/bin/bash
# Runs every example end to end. The examples exercise the public API only, so
# a failure here means a user-visible break even when the test suite is green.
set -e

# `tools/ensure_decimo.sh` has to have run: Linamo does not compile without
# decimo. It is normally taken from the pixi environment, leaving `temp/` empty.
if [[ ! -d temp ]]; then
    echo "decimo has not been resolved. Run 'pixi run decimo' first." >&2
    exit 1
fi
INCLUDES=(-I src -I temp)

for f in examples/*.mojo; do
    echo "=========================================="
    echo "Running: $f"
    echo "=========================================="
    pixi run mojo run "${INCLUDES[@]}" "$f"
done

echo ""
echo "=========================================="
echo "All examples ran successfully!"
echo "=========================================="
