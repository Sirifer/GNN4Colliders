date

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/configs/model_configs/"

configs=(  
  "run_3_ttH/v05/sb_yukawa_cp_abs_weights.yaml --abs"
)

counter=0

hours=12
time="${hours}:00:00"

for job in "${configs[@]}"
do
  sbatch --job-name="$job" --time="$time" /global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/jobs/run_job.sh "$job"
  ((counter++))
done

echo "Total jobs submitted: $counter"