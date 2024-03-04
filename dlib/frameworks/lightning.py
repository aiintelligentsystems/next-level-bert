import time

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_info


class dLitCheckpointLoadMixin:
    """
    Mixin that allows loading weights from checkpoints, even if they are partially missing.
    Errors are printed and handled gracefully.
    """

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()  # type: ignore
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)


class CUDAMetricsCallback(Callback):
    """
    Log CUDA stats. Adapted from https://github.com/Lightning-AI/lightning-GPT/blob/main/lightning_gpt/callbacks.py
    """

    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2**20
        start_time = getattr(self, "start_time", None)
        if start_time:
            time_since_last_validation = time.time() - self.start_time

            mean_max_memory = trainer.strategy.reduce(max_memory, reduce_op="mean")
            max_max_memory = trainer.strategy.reduce(max_memory, reduce_op=torch.max)

            time_since_last_validation = trainer.strategy.reduce(time_since_last_validation)

            rank_zero_info(f"Average time: {time_since_last_validation:.2f} seconds")
            rank_zero_info(f"Average Peak memory {mean_max_memory:.2f}MiB")
            rank_zero_info(f"Max Peak memory {max_max_memory:.2f}MiB")

            trainer.logger.log_metrics(
                {
                    "System/Avg. Peak CUDA memory": mean_max_memory,
                    "System/Max. Peak CUDA memory": max_max_memory,
                    "System/Avg. Training Time": time_since_last_validation,
                },
                step=trainer.fit_loop.epoch_loop._batches_that_stepped,
            )

    def root_gpu(self, trainer: "Trainer") -> int:
        return trainer.strategy.root_device.index
