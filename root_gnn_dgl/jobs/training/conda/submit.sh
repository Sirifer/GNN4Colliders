date

DIRECTORY="/global/cfs/projectdirs/atlas/joshua/root_gnn/root_gnn_dgl/configs/model_configs/"

configs=(  
  "configs/stats_all/ttH_CP_even_vs_odd.yaml"
  "configs/stats_all/ttH_CP_even_vs_odd_batch_size_2048.yaml"
  "configs/stats_all/ttH_CP_even_vs_odd_batch_size_4096.yaml"
  "configs/stats_all/ttH_CP_even_vs_odd_batch_size_8192.yaml"
)

counter=0

hours=12
time="${hours}:00:00"

for job in "${configs[@]}"
do
  sbatch --job-name="$job" --time="$time" /global/cfs/projectdirs/atlas/joshua/gnn/root_gnn_dgl/jobs/training/conda/run_job.sh "$job"
  ((counter++))
done

echo "Total jobs submitted: $counter"