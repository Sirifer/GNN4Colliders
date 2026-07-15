#!/bin/bash

ACCOUNT="${SLURM_ACCOUNT:-atlas}"

salloc --nodes 4 --qos interactive --time 04:00:00 --constraint gpu --account="$ACCOUNT" --gres=gpu:4
