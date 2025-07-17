FROM nvcr.io/nvidia/dgl:25.05-py3

WORKDIR /global/cfs/projectdirs/atlas/joshua/GNN4Colliders

LABEL maintainer.name="Joshua Ho"
LABEL maintainer.email="ho22joshua@berkeley.edu"

ENV LANG=C.UTF-8

# Install system dependencies: vim, OpenMPI, and build tools
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    wget lsb-release gnupg software-properties-common \
    vim \
    g++-11 gcc-11 libstdc++-11-dev \
    openmpi-bin openmpi-common libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python packages: mpi4py and jupyter
RUN pip install --no-cache-dir mpi4py jupyter uproot

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
i
# (Optional) Expose Jupyter port
EXPOSE 8888


