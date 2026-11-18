#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=hzg-Zsignificance-4class-v3-bkg-4cat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/significance_4class_v3_bkg_4cat-%j.out
#SBATCH --error=/scratchfs2/cms/huangsuyun/GNN4Colliders/logs/significance_4class_v3_bkg_4cat-%j.err

set -euo pipefail

REPO=/scratchfs2/cms/huangsuyun/GNN4Colliders
PYTHON=/publicfs/cms/user/huangsuyun/conda/envs/pytorch/bin/python
SCRIPT=${REPO}/tools/significance_4class_v3_bkg_4cat.py
RESULT_DIR=${REPO}/tools/results_4class_v3_bkg_4cat
RUN_RESULT=${RESULT_DIR}/job_${SLURM_JOB_ID:-manual}.txt

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MPLCONFIGDIR=${REPO}/cache/matplotlib

mkdir -p ${REPO}/logs ${MPLCONFIGDIR} ${RESULT_DIR}
cd ${REPO}

echo "=== HZgamma 4-class Bkg-MC scaled significance ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Python: ${PYTHON}"
echo "Script: ${SCRIPT}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-4}"
echo "Arguments: $*"

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-4} \
    ${PYTHON} ${SCRIPT} "$@" 2>&1 | tee "${RUN_RESULT}"

echo "Run output: ${RUN_RESULT}" | tee -a "${RUN_RESULT}"
echo "Final report: ${RESULT_DIR}/final_z.txt" | tee -a "${RUN_RESULT}"
echo "Category yields: ${RESULT_DIR}/category_yields.csv" | tee -a "${RUN_RESULT}"

echo "=== Done ==="
