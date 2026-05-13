#!/bin/bash
# macOS Gatekeeper blocks NumPy/PyTorch .so and .dylib files in the venv.
# This removes the quarantine attribute so they can load. Run once from repo root:
#   ./fix_venv_gatekeeper.sh
set -e
cd "$(dirname "$0")"
VENV="PoisonedRAG_env"
if [[ ! -d "$VENV" ]]; then
  echo "No $VENV directory found."
  exit 1
fi
echo "Removing quarantine attribute from $VENV (so NumPy/PyTorch can load)..."
xattr -dr com.apple.quarantine "$VENV"
echo "Done. Try running your code again (e.g. ./run_defense_demo.sh)."
echo "If you still see security dialogs, click 'Done' (do not click 'Move to Trash')."
