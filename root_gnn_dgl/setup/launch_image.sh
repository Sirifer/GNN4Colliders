podman-hpc run \
  -it \
  --mount type=bind,source=/pscratch/sd/j/joshuaho/,target=/pscratch/sd/j/joshuaho/ \
  --mount type=bind,source=/global/cfs/projectdirs/atlas/joshua/,target=/global/cfs/projectdirs/atlas/joshua/ \
  --rm \
  --network host \
  --gpu \
  --shm-size=32g \
  joshuaho/pytorch:1.0