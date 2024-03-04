#!/bin/bash
# please review the following environment variable settings depending on your setup
export WANDB_MODE=online
export WANDB_DATA_DIR=/home/mamba/.cache/wandb
export NLTK_DATA=/workspace/nltk_data
export TMPDIR=/home/mamba/.cache/tmp
export TRANSFORMERS_CACHE=/home/mamba/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/home/mamba/.cache/torch/sentence_transformers
set -e
echo $HF_DATASETS_CACHE
CHUNKING=256
ENCODER_BATCH_SIZE=4096
DEVICES=$CUDA_VISIBLE_DEVICES
/opt/conda/bin/python train.py \
    --data_preprocessing_only \
    --accelerator=gpu \
    --gpus 1 \ # don't change this even if you have more devices available. they will be used
    --dataset pile_books \
    --encoder_name all-MiniLM-L6-v2 \ # currently needs to be a model available via the sentence_transformers library
    --chunking $CHUNKING \ # choose your desired chunking strategy of the input texts. 0 corresponds to punctuation-based chunking
    --encoder_batch_size $ENCODER_BATCH_SIZE \
    
wait

