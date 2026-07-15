#!/bin/bash

ACCOUNT="${SLURM_ACCOUNT:-atlas}"

salloc --nodes=1 --ntasks=64 --cpus-per-task=1 --qos=interactive --time=04:00:00 --constraint=cpu --account="$ACCOUNT"
