---
name: root-gnn-dgl-env-setup
description: Use when the user asks to set up or validate the root_gnn_dgl runtime, such as running conda setup from setup/environment.yml, setup/test_setup.py, import ROOT checks, podman-hpc image setup, or the interactive allocation scripts in jobs/ before data prep, training, or inference.
---

# root-gnn-dgl-env-setup

Use this skill from the repo root before any stage run.

## Choose the runtime

- Use the conda environment in `setup/environment.yml` for `scripts/inference.py`. The repo README says inference needs PyROOT, and the podman image does not include ROOT.
- Use the `podman-hpc` image `joshuaho/pytorch:1.0` for training on Perlmutter when you want the containerized path.
- For parallel inference, make sure `mpi4py` is available. The README notes it is not listed in the conda environment requirements; `setup/Dockerfile` installs it in the container image.

## Conda path

```bash
cd setup
conda env create -f environment.yml
conda activate pytorch
cd ..
python setup/test_setup.py
python -c "import ROOT"
```

Run `setup/test_setup.py` from the repo root. It appends the current working directory to `sys.path` and checks imports in `scripts`, `root_gnn_base`, and `models`.

## Podman path

```bash
podman-hpc pull docker.io/joshuaho/pytorch:1.0
```

Or build locally:

```bash
cd setup
source build_image.sh
```

The helper `setup/launch_image.sh` mounts `/pscratch/sd/j/joshuaho/` and `/global/cfs/projectdirs/atlas/joshua/` into the container and then runs the given entrypoint.

## Interactive allocations

- `source jobs/interactive.sh` for one shared interactive GPU node.
- `source jobs/cpu.sh` for a CPU allocation that suits large prep loops.
- `source jobs/salloc.sh` for a multi-node GPU allocation.

## Runtime audit

- Use `nvidia-smi` before training on login or interactive nodes to confirm memory availability.
- Validate the basic repo imports with `python setup/test_setup.py`.
- Validate PyROOT explicitly with `python -c "import ROOT"`.
- For parallel inference, also validate `python -c "from mpi4py import MPI"`.
- Some repo scripts hard-code NERSC-style paths under `/global/cfs/projectdirs/atlas/joshua/...`. If running elsewhere, fix those paths before assuming the environment is valid.

Treat environment setup as passing only if:

- imports succeed
- the chosen runtime matches the stage you plan to run
- required site-specific paths exist
- GPU or CPU resources are actually available for the intended stage
