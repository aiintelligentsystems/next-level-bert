# NextLevelBERT

[![Docker Hub](https://img.shields.io/docker/v/konstantinjdobler/nlp-research-template/torch2.0.0-cuda11.8?color=blue&label=docker&logo=docker)](https://hub.docker.com/repository/docker/taczin/next_level/tags) ![License: MIT](https://img.shields.io/github/license/konstantinjdobler/nlp-research-template?color=green)

Code for [NextLevelBERT: Investigating Masked Language Modeling with Higher-Level Representations for Long Documents](https://arxiv.org/abs/2402.17682).

## Setup
It's recommended to use [`mamba`](https://github.com/mamba-org/mamba) to manage dependencies. `mamba` is a drop-in replacement for `conda` re-written in C++ to speed things up significantly (you can stick with `conda` though). To provide reproducible environments, we use `conda-lock` to generate lockfiles for each platform.

This code repository is based on the [NLP Research Template](https://github.com/konstantinjdobler/nlp-research-template). More details on setting up `mamba` and `conda-lock` can be found there.

### Docker

For a fully reproducible environment and running on HPC clusters, we provide pre-built docker images at [https://hub.docker.com/r/taczin/next_level/tags](https://hub.docker.com/r/taczin/next_level/tags). We also provide a `Dockerfile` that allows you to build new docker images with updated dependencies:

```bash
docker build --tag <username>/<imagename>:<tag> --platform=linux/<amd64/ppc64le> .
```

### Customize
You can activate the environment by running 
```
bash scripts/console.sh
```
which will start a docker container in an interactive session.
Before you can start successfully, you have to adapt the GPU devices and dataset mount paths in `scripts/console.sh`.

We are using Weights & Biases. To enable W&B, enter your `WANDB_ENTITY` and `WANDB_PROJECT` in [dlib/frameworks/wandb.py](dlib/frameworks/wandb.py).


## Preprocessing
To pretrain the model, the pretraining data first has to be preprocessed separately. You can run this via:

```
bash scripts/preprocess.sh
```
This processes the data with the all-MiniLM-L6-v2 sentence-transformers model, with a chunk size of 256 and an encoder model batch size of 4096.

## Pretraining

To start Next-Level pretraining, run:

```
bash scripts/pretrain.sh
```
We cannot provide the books3 dataset we used for pretraining and it was taken off most easily accessible platforms like the huggingface hub due to license controversy. If you don't have a version of the dataset available you can try substituting it by other book-based data, e.g., project Gutenberg. We have not tested data with more short-ranged depencencies, but it might also work. If you try it out, we would love to hear about your results.

## Downstream Evaluation

First run preprocessing for the downstream dataset you want. You can edit the dataset in the script file.
```
bash scripts/preprocess_downstream.py
```
Then run fine-tuning and evaluation by:
```
bash scripts/downstream.sh
```
Note that the quality dataset needs to be downloaded first. (See `scripts/download_quality.sh`.) The BookSum dataset is used for zero-shot embedding quality evaluation and is not fine-tuned on.

## In the Future
We are currently working on making model checkpoints available and providing an easier way of using the model for inference. Stay tuned.
