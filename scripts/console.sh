#!/bin/bash

set e

# Check if the current directory is named "scripts"
current_directory=$(basename "$PWD")

if [ "$current_directory" == "scripts" ]; then
  echo "Error: This script should be called from the root of the project."
  echo "Example: bash ./scripts/console.sh"
  exit 1
fi
docker run -it \
    --gpus \"device=<your_cuda_device_idx>\" \ #e.g. --gpus \"device=0,1\" 
    --ipc host \
    --rm \
    --user "$(id -u):$(id -g)" \
    --env WANDB_API_KEY \
    --env HF_DATASETS_CACHE='/home/mamba/.cache/' \
    -v </path/to/large/folder>:/home/mamba/.cache \
    -v </path/to/the_pile_books3>:/home/mamba/.cache/huggingface/datasets/the_pile_books3/base \
    -v </path/to/booksum>:/home/mamba/.cache/huggingface/datasets/booksum/base \
    -v </path/to/muld/movie_scripts>:/home/mamba/.cache/huggingface/datasets/ghomasHudson___muld/base \
    -v "$(pwd)":/workspace \
    -w /workspace \
    taczin/next_level:latest \
    bash
