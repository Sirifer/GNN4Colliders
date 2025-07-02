#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH -q regular
#SBATCH --mail-user=ho22joshua@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 05:00:00
#SBATCH -A atlas
#SBATCH -o /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/jobs/slurm/%j.out # STDOUT


CONFIG="$*"

echo "Executing command: $CONFIG"

cd /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/

eval "$(conda shell.bash hook)"
conda init bash
conda activate /global/homes/j/joshuaho/.conda/envs/dgl

# Run the Python script and capture the output
MASTER_PORT=$(python /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/scripts/find_free_port.py)

# export MASTER_ADDR=$(hostname)
export MASTER_PORT=$MASTER_PORT

# Dynamically get the hostname of the first node in the allocation to use as MASTER_ADDR
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Debugging: Print the master address and port
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

TORCHRUN_ARGUMENTS="--nnodes=$SLURM_NNODES \
                    --nproc-per-node=$SLURM_GPUS_ON_NODE \
                    --rdzv-id=$SLURM_JOB_ID \
                    --rdzv-backend=c10d \
                    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT"

GPUS=$(( $SLURM_NNODES * 4 ))

echo GPUS: $GPUS

srun --gpus=$GPUS \
    --ntasks-per-node=1 \
    /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/launch_image.sh \
    "--entrypoint /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/jobs/run_multinode_2.sh" \
    $TORCHRUN_ARGUMENTS \
    $CONFIG