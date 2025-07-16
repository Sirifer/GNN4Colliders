nsys profile \
  -o /pscratch/sd/j/joshuaho/full_stats_profile_1_gpu_batch_size_1028 \
  --capture-range=cudaProfilerApi \
  --duration=100 \
  --force-overwrite true \
  --trace=nvtx \
  --cudabacktrace=all \
  python scripts/training_script.py --config configs/stats_all/ttH_CP_even_vs_odd.yaml --preshuffle --nocompile --lazy --restart --profile

nsys profile \
  -o /pscratch/sd/j/joshuaho/full_stats_profile_1_gpu_batch_size_2048 \
  --capture-range=cudaProfilerApi \
  --duration=100 \
  --force-overwrite true \
  --trace=nvtx \
  --cudabacktrace=all \
  python scripts/training_script.py --config configs/stats_all/ttH_CP_even_vs_odd_batch_size_2048.yaml --preshuffle --nocompile --lazy --restart --profile

nsys profile \
  -o /pscratch/sd/j/joshuaho/full_stats_profile_1_gpu_batch_size_4096 \
  --capture-range=cudaProfilerApi \
  --duration=100 \
  --force-overwrite=true \
  --trace=nvtx \
  --cudabacktrace=all \
  python scripts/training_script.py --config configs/stats_all/ttH_CP_even_vs_odd_batch_size_4096.yaml --preshuffle --nocompile --lazy --restart --profile

nsys profile \
  -o /pscratch/sd/j/joshuaho/full_stats_profile_1_gpu_batch_size_8192 \
  --capture-range=cudaProfilerApi \
  --duration=100 \
  --force-overwrite true \
  --trace=nvtx \
  --cudabacktrace=all \
  python scripts/training_script.py --config configs/stats_all/ttH_CP_even_vs_odd_batch_size_8192.yaml --preshuffle --nocompile --lazy --restart --profile
