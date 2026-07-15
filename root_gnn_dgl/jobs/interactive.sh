#!/bin/bash

ACCOUNT="${SLURM_ACCOUNT:-atlas}"

salloc --nodes 1 --qos shared_interactive --time 04:00:00 --constraint gpu --account="$ACCOUNT" --gres=gpu:1
