#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --mail-user=ho22joshua@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 15:00:00
#SBATCH -A atlas
#SBATCH -o /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/jobs/slurm/%j.out # STDOUT

ARGUEMENTS="$*"

echo "Arguements: $ARGUEMENTS"
echo "launching image"
source launch_image.sh "--entrypoint /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/jobs/run_job_image.sh" $ARGUEMENTS