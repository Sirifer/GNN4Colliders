nsys profile \
  -o /pscratch/sd/j/joshuaho/my_profile_report_1_gpu_batch_size_1028 \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --force-overwrite true \
  --trace=nvtx \
  --cudabacktrace=all \
  python scripts/training_script.py --config configs/stats_100K/ttH_CP_even_vs_odd.yaml --preshuffle --nocompile --lazy --restart --profile

nsys profile \
  -o /pscratch/sd/j/joshuaho/my_profile_report_1_gpu_batch_size_2048 \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --force-overwrite true \
  --trace=nvtx \
  --cudabacktrace=all \
  python scripts/training_script.py --config configs/stats_100K/ttH_CP_even_vs_odd_batch_size_2048.yaml --preshuffle --nocompile --lazy --restart --profile

nsys profile \
  -o /pscratch/sd/j/joshuaho/my_profile_report_1_gpu_batch_size_4096 \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --force-overwrite true \
  --trace=nvtx \
  --cudabacktrace=all \
  python scripts/training_script.py --config configs/stats_100K/ttH_CP_even_vs_odd_batch_size_4096.yaml --preshuffle --nocompile --lazy --restart --profile

nsys profile \
  -o /pscratch/sd/j/joshuaho/my_profile_report_1_gpu_batch_size_8192 \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --force-overwrite true \
  --trace=nvtx \
  --cudabacktrace=all \
  python scripts/training_script.py --config configs/stats_100K/ttH_CP_even_vs_odd_batch_size_8192.yaml --preshuffle --nocompile --lazy --restart --profile
