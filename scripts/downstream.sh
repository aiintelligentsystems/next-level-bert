#!/bin/bash
# please review the following environment variable settings depending on your setup
export WANDB_MODE=online
export WANDB_DATA_DIR=/home/mamba/.cache/wandb
export NLTK_DATA=/workspace/nltk_data
export TMPDIR=/home/mamba/.cache/tmp
export TRANSFORMERS_CACHE=/home/mamba/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/home/mamba/.cache/torch/sentence_transformers
set -e
CHUNKING=256
NUM_DEVICES=$(nvidia-smi  -L | wc -l)
echo $NUM_DEVICES
/opt/conda/bin/python train.py \
    --accelerator=gpu \
    --gpus $NUM_DEVICES \
    --eval_dataset quality \
    --encoder_name all-MiniLM-L6-v2 \
    --chunking $CHUNKING \
    --loss_func smoothl1 \
    --batch_size_per_device 64 \
    --lr 1e-4 \
    --max_epochs 20 \
    --gas 2 \
    --masking_probability 0.15 \
    --val_frequency 2500 \
    --model_log_frequency 2500 \
    --workers 8 \
    --compile \
    --val_batches 500 \
    
wait