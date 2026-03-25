#!/bin/bash
#SBATCH -N 1
#SBATCH -C "gpu&hbm80g"
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -A atlas
#SBATCH -o /global/cfs/projectdirs/atlas/joshua/gnn/root_gnn_dgl/jobs/slurm/%j.out # STDOUT

ARGUEMENTS="$*"

echo "Arguements: $ARGUEMENTS"

## create a launch image script
source "/global/cfs/projectdirs/atlas/joshua/gnn/root_gnn_dgl/setup/launch_image.sh" "/global/cfs/projectdirs/atlas/joshua/gnn/root_gnn_dgl/jobs/training/podman/run_job_image.sh" $ARGUEMENTS
