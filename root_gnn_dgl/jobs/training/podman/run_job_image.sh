#!/bin/bash

CONFIG=$1
shift
# Store any other potential arguments safely
OTHER_ARGUEMENTS=("$@")

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/gnn/root_gnn_dgl/"

cd $DIRECTORY

# Use a bash array to build the command and its arguments
# Each element in the () is a separate argument.
COMMAND_ARGS=(
    "$DIRECTORY/scripts/training_script.py"
    "${OTHER_ARGUEMENTS[@]}"
    "--preshuffle"
    "--nocompile"
    "--lazy"
    "--config"
    "$DIRECTORY$CONFIG"
)

echo "Running my script now"
# Using "@" in quotes expands the array correctly
echo "Executing: python -u ${COMMAND_ARGS[@]}"

# The "${COMMAND_ARGS[@]}" syntax ensures each element is passed as a distinct argument
python -u "${COMMAND_ARGS[@]}"

echo "Done"