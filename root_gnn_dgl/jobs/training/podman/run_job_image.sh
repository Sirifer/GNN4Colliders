#!/bin/bash

CONFIG=$1
shift
# Store any other potential arguments safely
OTHER_ARGUEMENTS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

# Use a bash array to build the command and its arguments
# Each element in the () is a separate argument.
COMMAND_ARGS=(
    "$REPO_ROOT/scripts/training_script.py"
    "${OTHER_ARGUEMENTS[@]}"
    "--preshuffle"
    "--nocompile"
    "--lazy"
    "--config"
    "$CONFIG"
)

echo "Running my script now"
# Using "@" in quotes expands the array correctly
echo "Executing: python -u ${COMMAND_ARGS[@]}"

# The "${COMMAND_ARGS[@]}" syntax ensures each element is passed as a distinct argument
python -u "${COMMAND_ARGS[@]}"

echo "Done"
