#!/bin/bash

# Arguments
config=$1   # YAML configuration file
dataset=$2  # Dataset name
CHUNKS=$3  # Number of chunks

args=$4     # Additional arguments to pass to prep_data.py

# Prepare the initial command for chunk 0
command="scripts/prep_data.py --dataset $dataset --shuffle_mode --chunk 0 --config $config $args"
echo "$command"
python $command

# Loop through all chunks
for ((i=0; i<$CHUNKS; i++)); do
  command="scripts/prep_data.py --dataset $dataset --shuffle_mode --chunk $i --config $config $args"
  echo "$command"
  python $command
done