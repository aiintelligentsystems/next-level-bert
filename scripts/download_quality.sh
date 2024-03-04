#!/bin/bash

TARGET_DIR=/home/mamba/.cache/huggingface/datasets/quality/raw/
if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p $TARGET_DIR
fi
cd $TARGET_DIR
wget https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.train
wget https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
wget https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test