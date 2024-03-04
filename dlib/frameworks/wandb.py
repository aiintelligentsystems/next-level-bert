import os
import re
import subprocess
import threading
from pathlib import Path

import pytorch_lightning as L
import wandb
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger

from ..frameworks.pytorch import get_rank

WANDB_PROJECT = "next-level-lms"
WANDB_ENTITY = "aiis-nlp"

if WANDB_ENTITY == "<your entity>" or WANDB_PROJECT == "<your project>":
    logger.warning(
        "dlib error: You need to specify WANDB_ENTITY and WANDB_PROJECT in dlib/frameworks/wandb.py when using the wandb module."
    )


class MyWandbLogger(WandbLogger):
    def _scan_and_log_checkpoints(self, checkpoint_callback) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = {
            checkpoint_callback.last_model_path: checkpoint_callback.current_score,
            checkpoint_callback.best_model_path: checkpoint_callback.best_model_score,
            **checkpoint_callback.best_k_models,
        }
        checkpoints = sorted(
            (Path(p).stat().st_mtime, p, s) for p, s in checkpoints.items() if Path(p).is_file()
        )
        checkpoints = [
            c
            for c in checkpoints
            if c[1] not in self._logged_model_time.keys() or self._logged_model_time[c[1]] < c[0]
        ]

        # log iteratively all new checkpoints
        for t, p, s in checkpoints:
            metadata = {
                "score": s,
                "original_filename": Path(p).name,
                "ModelCheckpoint": {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            artifact = wandb.Artifact(
                name=f"model-{self.experiment.id}", type="model", metadata=metadata
            )
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)

            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t


class WandbCleanupDiskAndCloudSpaceCallback(Callback):
    def __init__(self, cleanup_local=True, cleanup_online=True, size_limit=0, backoff=10) -> None:
        super().__init__()
        self.cleanup_local = cleanup_local
        self.cleanup_online = cleanup_online
        self.size_limit = size_limit
        self.backoff = backoff
        self.counter = 0

    @rank_zero_only
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.counter < self.backoff:
            self.counter += 1
            return
        else:
            self.counter = 0
        # run = wandb.run  # type: ignore
        run = trainer.logger.experiment  # type: ignore

        # Delete outdated online artifacts
        if self.cleanup_online:  # BUG: this doesn't work....
            if getattr(run, "logged_artifacts", None) is not None:
                for artifact in run.logged_artifacts():
                    aliases = [x["alias"] for x in artifact._attrs["aliases"]]
                    if "best" not in aliases and "keep" not in aliases:
                        logger.info(f"Deleting outdated artifact with aliases {aliases}")
                        artifact.delete()
            else:
                logger.error("wandb run has no logged artifacts")

        # Free up local wandb cache (This is often A LOT of memory)
        if self.cleanup_local:
            logger.info("Starting wandb artifact cache cleanup timeout")
            cache_cleanup_callback = lambda: subprocess.run(  # noqa: E731
                ["wandb", "artifact", "cache", "cleanup", f"{self.size_limit}GB"]
            )
            timer = threading.Timer(
                120.0, cache_cleanup_callback
            )  # Delay cleanupcall to avoid cleaning a temp file from the ModelCheckpoint hook that is needed to upload current checkpoint
            timer.start()


def check_for_wandb_checkpoint_and_download_if_necessary(
    checkpoint_path: str,
    wandb_run_instance,
    wandb_entity=None,
    wandb_project=None,
    suffix="/model.ckpt",
) -> str:
    """
    Checks the provided checkpoint_path for the wandb regex r\"wandb:.*\".
    If matched, download the W&B artifact indicated by the id in the provided string and return its path.
    If not, just returns provided string.
    """
    wandb_model_id_regex = r"wandb:.*"
    if re.search(wandb_model_id_regex, checkpoint_path):
        if get_rank() == 0:
            logger.info("Downloading W&B checkpoint...")
        wandb_model_id = checkpoint_path.split(":")[1]
        model_tag = (
            checkpoint_path.split(":")[2] if len(checkpoint_path.split(":")) == 3 else "latest"
        )

        """
        Only the main process should download the artifact in DDP. We add this environment variable as a guard. 
        This works only if this function is called first on the main process.
        """
        if os.environ.get(f"DLIB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"):
            checkpoint_path = os.environ[f"DLIB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"]
        else:
            artifact = wandb_run_instance.use_artifact(
                f"{wandb_entity or WANDB_ENTITY}/{wandb_project or WANDB_PROJECT}/model-{wandb_model_id}:{model_tag}"
            )
            checkpoint_path = artifact.download() + suffix
            logger.info(f"Path of downloaded W&B artifact: {checkpoint_path}")
            os.environ[f"DLIB_ARTIFACT_PATH_{wandb_model_id}_{model_tag}"] = checkpoint_path
    return checkpoint_path


def check_checkpoint_path_for_wandb(checkpoint_path: str):
    wandb_model_id_regex = r"wandb:.*"
    if re.search(wandb_model_id_regex, checkpoint_path):
        wandb_model_id = checkpoint_path.split(":")[1]
        return wandb_model_id
    return None


def patch_transformers_wandb_callback():
    """Patch the transformers wandb integration to use the run id instead of the run name to name artifacts."""
    import numbers
    import tempfile

    from transformers.integrations import WandbCallback

    def patched_on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if (
            self._log_model in ("end", "checkpoint")
            and self._initialized
            and state.is_world_process_zero
        ):
            from transformers import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                logger.info("Logging model artifacts. ...")
                # Always use run id
                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (True or args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    WandbCallback.on_train_end = patched_on_train_end
