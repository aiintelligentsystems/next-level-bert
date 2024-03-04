import contextlib
import os
from time import sleep

import torch
from loguru import logger


def get_num_devices(gpu_specifier):
    num_gpus = 1
    if gpu_specifier == -1:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("GPUs requested but none found")
    elif isinstance(gpu_specifier, list):
        num_gpus = len(gpu_specifier)
    elif isinstance(gpu_specifier, int):
        num_gpus = gpu_specifier
    return int(num_gpus)


def get_effective_batch_size_per_step(gpu_specifier, batch_size: int):
    multiplier = 1
    if gpu_specifier == -1:
        multiplier = torch.cuda.device_count()
        if multiplier == 0:
            print("GPUs requested but none found")
    elif isinstance(gpu_specifier, list):
        multiplier = len(gpu_specifier)
    elif isinstance(gpu_specifier, int):
        multiplier = gpu_specifier
    return int(multiplier * batch_size)


def set_torch_file_sharing_strategy_to_system(worker_id: int = 0) -> None:
    """
    When having many workers for dataloaders / many tensors per batch, torch uses file descriptors to share data between processes.
    UNIX systems have upper limits for the number of open file descriptors allowed, given enough workers / tensors this limit will be reached and the process will be killed.
    https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
    """
    torch.multiprocessing.set_sharing_strategy("file_system")


@contextlib.contextmanager
def main_process_first(description="", active=True, time_buffer_after_main: bool | int = True):
    """
    Context manager that executes the wrapped code on the main process first and then on all other processes. Useful for e.g. dataset preprocessing.
    Inspiration from Huggingface: https://github.com/huggingface/transformers/pull/12351/files
    """
    if torch.distributed.is_available() and active:
        local_rank = get_rank()
        try:
            if local_rank > 0:
                logger.info(f"Process {local_rank} | {description} | Waiting for main process...")
                torch.distributed.barrier()
            yield
        finally:
            if local_rank == 0:
                logger.info(
                    f"Main process | {description} | Done. Executing on parallel processes now..."
                )
                torch.distributed.barrier()
                if time_buffer_after_main:
                    time_buffer_after_main = (
                        time_buffer_after_main if isinstance(time_buffer_after_main, int) else 30
                    )
                    sleep(time_buffer_after_main)  # Give other processes time to catch up
    else:
        yield


def get_rank() -> int:
    if not torch.distributed.is_available():
        return 0  # Training on CPU
    if not torch.distributed.is_initialized():
        rank = os.environ.get("LOCAL_RANK")  # from pytorch-lightning
        if rank is not None:
            return int(rank)
        else:
            return 0
    else:
        return torch.distributed.get_rank()
