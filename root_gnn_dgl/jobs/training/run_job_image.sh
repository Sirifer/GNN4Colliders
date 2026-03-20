#!/bin/bash

CONFIG=$1
shift
ARGUEMENTS="$*"

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/GNN4Colliders/root_gnn_dgl/"
COMMAND="$DIRECTORY"scripts/training_script.py $ARGUEMENTS --preshuffle --nocompile --lazy --config $DIRECTORY$CONFIG

echo "Running my script now"
echo $COMMAND
python -u $COMMAND
echo "Done"