FROM nvcr.io/nvidia/dgl:25.05-py3

WORKDIR /workspace

LABEL maintainer.name="Joshua Ho"
LABEL maintainer.email="ho22joshua@berkeley.edu"

ENV LANG=C.UTF-8

# System deps (with CA certs for HTTPS downloads)
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    wget curl ca-certificates lsb-release gnupg software-properties-common \
    vim \
    g++-11 gcc-11 libstdc++-11-dev \
    openmpi-bin openmpi-common libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir mpi4py jupyter uproot

EXPOSE 8888