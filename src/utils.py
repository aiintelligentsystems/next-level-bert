from typing import Callable, Dict, Any
import datasets
import torch
import os
import torch.multiprocessing as mp
from src.model import CustomizedSBERT
import datetime

cache_path = os.environ.get("HF_DATASETS_CACHE")


def delete_dataset(dataset):
    cached_files = [cache_file["filename"] for cache_file in dataset.cache_files]
    del dataset
    for cached_file in cached_files:
        if os.path.exists(cached_file):
            os.remove(cached_file)


def encoder_fn(examples, encoder, batch_size, rank=0):
    embeddings = encoder.encode(
        examples["chunks"],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        convert_to_tensor=False,
        device=f"cuda:{rank}",
        normalize_embeddings=False,
    )
    return {"embeddings": embeddings}


def dataset_map_multi_worker(
    idx, dataset, map_fn: Callable, kwargs_dict: Dict[str, Any], *args
) -> datasets.Dataset:
    second_level_tokenizer = CustomizedSBERT(
        f'sentence-transformers/{kwargs_dict["fn_kwargs"].pop("encoder_name")}',
        device=f"cuda:{idx}",
    )
    kwargs_dict["fn_kwargs"]["rank"] = idx
    kwargs_dict["fn_kwargs"]["encoder"] = second_level_tokenizer
    save_path = kwargs_dict.pop("save_file_path")
    try:
        setup(idx, torch.cuda.device_count())
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        dataset, _ = dataset.map(map_fn, *args, **kwargs_dict), delete_dataset(dataset)
        dataset = dataset.remove_columns(list(set(dataset.column_names) - set(["embeddings"])))
        dataset.set_format(type="torch", columns=["embeddings"], output_all_columns=True)
        dataset.save_to_disk(f"{save_path}sentences")
        return
    ds_shard_filepaths = [
        os.path.join(save_path, f"{dataset._fingerprint}_subshard_{w}.cache")
        for w in range(0, world_size)
    ]
    print(f"\tworker {rank} saving sub-shard to {ds_shard_filepaths[rank]}")
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )
    ds_shard = ds_shard.map(map_fn, *args, **kwargs_dict)
    ds_shard.save_to_disk(f"{ds_shard_filepaths[rank]}")
    print("rank", rank, "saving:", ds_shard_filepaths[rank])
    torch.distributed.barrier()
    if idx not in [-1, 0]:
        torch.distributed.barrier()
    else:
        full_dataset = datasets.concatenate_datasets(
            [datasets.load_from_disk(p) for p in ds_shard_filepaths]
        )
        full_dataset = full_dataset.remove_columns(
            list(set(full_dataset.column_names) - set(["embeddings"]))
        )
        full_dataset.set_format(type="torch", columns=["embeddings"], output_all_columns=True)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        full_dataset.save_to_disk(f"{save_path}sentences")
        del full_dataset
        torch.distributed.barrier()

    print("rank", rank, "deleting:", ds_shard_filepaths[rank])
    delete_dataset(ds_shard)
    cleanup()


def spawn_processes_helper(dataset, *args, **kwargs):
    map_fn = encoder_fn
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        dataset_map_multi_worker(0, dataset, map_fn, kwargs, *args)
    else:
        mp.spawn(
            dataset_map_multi_worker,
            args=(dataset, map_fn, kwargs, *args),
            nprocs=world_size,
            join=True,
        )


def setup(rank, world_size):
    if world_size <= 1:
        return
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=18000)
    )


def cleanup():
    torch.distributed.destroy_process_group()
