import dataclasses
import os
import time
from pathlib import Path

import torch
import wandb
from dargparser import dargparse
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import (
    LightningEnvironment,
    SLURMEnvironment,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from loguru import logger


from dlib.frameworks.lightning import CUDAMetricsCallback
from dlib.frameworks.pytorch import (
    get_effective_batch_size_per_step,
    get_num_devices,
    get_rank,
    set_torch_file_sharing_strategy_to_system,
)
from dlib.frameworks.wandb import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    WandbCleanupDiskAndCloudSpaceCallback,
    check_checkpoint_path_for_wandb,
    check_for_wandb_checkpoint_and_download_if_necessary,
)
from dlib.utils import wait_for_debugger
from src.data_loading import NextLevelDatamodule
from src.data_loading_downstream import DownstreamDatamodule
from src.helpers import infer_batch_size_per_device
from src.model import NextLevelLM, CustomizedSBERT, DownstreamModel
from src.baseline_models import PretrainedLongformer, PretrainedSBERT, AvgSBERTSentences

from arguments import TrainingArgs, MiscArgs


@logger.catch(reraise=True)
def main(parsed_arg_groups: tuple[TrainingArgs, MiscArgs]):
    current_process_rank = get_rank()
    args, misc_args = parsed_arg_groups

    ################ Apply fixes ##############
    if misc_args.too_many_open_files_fix:
        logger.info("Setting torch sharing strategy to 'file_system'")
        set_torch_file_sharing_strategy_to_system()
    if args.evaluate_downstream:
        args.resume_training = False
    if current_process_rank == 0 and args.debug:
        wait_for_debugger()
    ############# Seed & print args ##############
    misc_args.seed = seed_everything(workers=True, seed=misc_args.seed)
    current_process_rank = get_rank()
    if current_process_rank == 0:
        for arg_group in parsed_arg_groups:
            logger.info(arg_group)

    ############# Construct W&B Logger ##############
    if misc_args.offline or misc_args.fast_dev_run:  # or args.data_preprocessing_only:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_extra_args = dict(
        name=misc_args.wandb_run_name,
    )
    if (
        args.checkpoint_path
        and args.resume_training
        and check_checkpoint_path_for_wandb(args.checkpoint_path)
    ):
        logger.info("Resuming training from W&B")
        wandb_extra_args = dict(
            id=check_checkpoint_path_for_wandb(args.checkpoint_path), resume="must"
        )  # resume W&B run
    else:
        args.resume_training = False

    wandb_logger = WandbLogger(
        project=misc_args.wandb_project or WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model=not args.evaluate_downstream,
        tags=misc_args.wandb_tags,
        **wandb_extra_args,
    )

    # pick right batch size for preprocessing. assumes all GPUs are of the same type
    x, y = torch.cuda.mem_get_info(torch.device("cuda:0"))
    y = y // 1024**2
    if y < 34000:
        args.encoder_batch_size = 2048
    elif y < 50000 and y > 34000:
        args.encoder_batch_size = 4096
    elif y < 90000 and y > 50000:
        args.encoder_batch_size = 8192
    if args.encoder_name == "all-mpnet-base-v2":
        args.encoder_batch_size = args.encoder_batch_size // 64
    elif args.encoder_name == "all-roberta-large-v1":
        args.encoder_batch_size = args.encoder_batch_size // 32
        print(f"New batch size: {args.encoder_batch_size}")
    elif args.encoder_name == "all-distilroberta-v1":
        args.encoder_batch_size = args.encoder_batch_size // 32

    if args.chunking == 512:
        args.encoder_batch_size = args.encoder_batch_size // 4
    logger.info(f"New batch size: {args.encoder_batch_size}")
    #### Calculate effective batch size / gradient accumulation steps ####
    ACCELERATOR = args.accelerator.upper()
    num_devices = get_num_devices(args.devices)
    if args.effective_batch_size:
        logger.info(
            f"Trying to auto-infer settings for effective batch size {args.effective_batch_size}..."
        )
        (
            args.batch_size_per_device,
            args.gradient_accumulation_steps,
            effective_batch_size_per_step,
        ) = infer_batch_size_per_device(
            num_devices, args.effective_batch_size, args.batch_size_per_device
        )

        logger.info(
            f"Using effective batch size {args.effective_batch_size}"
            f"with {num_devices} {ACCELERATOR}s, "
            f"{args.batch_size_per_device} batch size per {ACCELERATOR} and "
            f"{args.gradient_accumulation_steps} gradient accumulation steps."
        )
    else:
        effective_batch_size_per_step = get_effective_batch_size_per_step(
            args.devices, args.batch_size_per_device
        )  # does not take accumulation into account
        args.effective_batch_size = effective_batch_size_per_step * args.gradient_accumulation_steps
        logger.info(
            f"Effective batch size {args.effective_batch_size} based on specified args"
            f"{num_devices} {ACCELERATOR}s, "
            f"{args.batch_size_per_device} batch size per {ACCELERATOR} and"
            f"{args.gradient_accumulation_steps} gradient accumulation steps."
        )

    for arg_group in parsed_arg_groups:
        wandb_logger.log_hyperparams(dataclasses.asdict(arg_group))

    if not args.encoder_name == "nomic":
        second_level_tokenizer = CustomizedSBERT(
            f"sentence-transformers/{args.encoder_name}", device="cuda"
        )
    else:
        second_level_tokenizer = CustomizedSBERT(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True, device="cuda"
        )
    first_level_tokenizer = second_level_tokenizer.tokenizer
    for param in second_level_tokenizer.parameters():
        param.requires_grad = False

    checkpoint_callback = ModelCheckpoint(
        filename="snap-{step}-ksamples-{progress/ksamples:.2f}-loss-{val/total_loss:.2f}",
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        # for pretrain save checkpoint according to model_log_frequency
        every_n_train_steps = args.model_log_frequency if not args.evaluate_downstream else None,
        # for downstream save after every validation epoch
        every_n_epochs = 1 if args.evaluate_downstream else None,
        save_on_train_epoch_end = False if args.evaluate_downstream else None,
        save_top_k=1,
    )
    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(
        cleanup_local=True, cleanup_online=False, size_limit=20
    )

    ################# Construct model ##############
    # Resume from checkpoint if specified
    if args.checkpoint_path:
        args.checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            args.checkpoint_path, wandb_logger.experiment
        )

        print("Resuming from", args.checkpoint_path)
        model = NextLevelLM.load_from_checkpoint(
            args.checkpoint_path,
            second_level_tokenizer=second_level_tokenizer,
        )
        if args.evaluate_downstream:
            model = DownstreamModel(
                args, misc_args, second_level_tokenizer, model
            )
    else:
        if args.model_name_or_path == "NextLevelLM":
            model = NextLevelLM(
                training_args=args,
                second_level_tokenizer=second_level_tokenizer,
                dropout=args.dropout,
                num_layers=args.num_layers,
                nhead=args.nhead,
            )
        elif args.model_name_or_path == "PretrainedLongformer":
            logger.info("You are loading a pretrained model. Only the prediction loop will run.")
            model = PretrainedLongformer(args, misc_args)
        elif args.model_name_or_path == "PretrainedSBERT":
            logger.info("You are loading a pretrained model. Only the prediction loop will run.")
            model = PretrainedSBERT(args, misc_args)
        elif args.model_name_or_path == "AvgSBERTSentences":
            logger.info("You are loading a pretrained model. Only the prediction loop will run.")
            model = AvgSBERTSentences(args, misc_args, second_level_tokenizer)
        else:
            raise ValueError("Wrong model name. Choose from NextLevelLM or PretrainedLongformer")

    if current_process_rank == 0:
        model.on_train_start = lambda: logger.info(
            f"Max training epochs: {args.max_epochs} | "
            f"Validation Frequency: {args.val_frequency} | "
            f"Model Log Frequencey: {args.model_log_frequency} | "
            f"Effective batch size: {args.effective_batch_size}"
        )
    

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")
    if args.evaluate_downstream:
        args.compile = False
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."  # noqa: E501
                "Please install torch >= 2.0 or disable compile."
            )
    #################### Construct dataloaders & trainer #################
    if not args.evaluate_downstream:
        if (args.model_name_or_path == "NextLevelLM") or (
            args.model_name_or_path == "AvgSBERTSentences"
        ):
            if isinstance(args.chunking, int):
                dm = NextLevelDatamodule(
                    args, misc_args, second_level_tokenizer, first_level_tokenizer, model
                )
            else:
                logger.error("Invalid chunking strategy!")
        else:
            raise ValueError("No suitable data module found for pretraining.")
    else:
        dm = DownstreamDatamodule(
            args, misc_args, second_level_tokenizer, first_level_tokenizer, model
        )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, wandb_disk_cleanup_callback, lr_monitor]
    if args.accelerator == "gpu":
        callbacks.append(CUDAMetricsCallback())

    # "smart" DDP skipping the find_unused_parameters step - slightly faster
    distributed_strategy = (
        DDPStrategy(find_unused_parameters=False)
        if args.accelerator == "gpu" and args.distributed_strategy == "ddp_smart"
        else args.distributed_strategy
    )
    wandb_logger.watch(model, log="gradients", log_freq=10 if args.evaluate_downstream else 500, log_graph=False)
    plugins = []
    if SLURMEnvironment.detect():
        logger.info("Disabling SLURMEnvironment (we use lightning's native DDP launcher)")
        plugins.append(LightningEnvironment())

    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor=args.early_stopping_metric, patience=args.early_stopping_patience, verbose=True
        )
        callbacks.append(early_stopping_callback)

    # Initialize trainer
    trainer = Trainer(
        num_sanity_val_steps=2,
        max_epochs=args.max_epochs,
        limit_val_batches=args.val_batches,
        val_check_interval=args.val_frequency,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=distributed_strategy,
        logger=wandb_logger,
        deterministic=misc_args.force_deterministic,
        callbacks=callbacks,
        plugins=plugins,
        enable_checkpointing=True,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )

    if current_process_rank == 0:
        logger.info(
            f"Total epochs: {args.max_epochs} | "
            f"LR warmup steps: {args.lr_warmup} | "
            f"Validation Frequency: {args.val_frequency} | "
            f"Model Log Frequency: {args.model_log_frequency} | "
            f"Effective batch size: {args.effective_batch_size} | "
            f"Micro batch size (per device and forward pass): {args.batch_size_per_device} | "
            f"Gradient accumulation steps: {args.gradient_accumulation_steps} | "
        )
    do_fit_model = not args.only_eval and ((not args.evaluate_downstream) or (args.evaluate_downstream and dm.dataset.needs_finetuning))
    if do_fit_model:
        ########### Start val & train loop ###########
        if args.val_before_training and not args.resume_training:
            logger.info(f"Rank {current_process_rank} | Validation before training...")
            val_result = trainer.validate(model, dm)
            print(val_result)
            if args.only_eval:
                exit(0)

        logger.info(f"Rank {current_process_rank} | Starting training...")
        trainer.fit(model, dm, ckpt_path=args.checkpoint_path if args.resume_training else None)
        if trainer.interrupted and SLURMEnvironment.detect():
            logger.error(
                "Detected keyboard interrupt, not trying to save latest checkpoint right now because we detected SLURM and do not want to drain the node..."
            )
        else:
            if trainer.interrupted:
                logger.warning("Detected keyboard interrupt, trying to save latest checkpoint...")
            if not args.evaluate_downstream:
                logger.info("Trying to save checkpoint....")
                save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
                trainer.save_checkpoint(save_path)

                if current_process_rank == 0:
                    logger.info("Collecting PL checkpoint for wandb...")
                    artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}", type="model")
                    artifact.add_file(save_path, name="model.ckpt")
                    logger.info("Pushing to wandb...")
                    aliases = ["train_end", "latest"]
                    wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

                    logger.success("Saving finished!")
        # load checkpoint with best validation loss before testing
        args_ckpt = {"checkpoint_path": trainer.checkpoint_callback.best_model_path}
        if model.__class__.__name__ == "DownstreamModel":
            args_ckpt["second_level_tokenizer"] = second_level_tokenizer
        model = model.__class__.load_from_checkpoint(
            **args_ckpt
        )
    # new trainer with maximum one GPU
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        accelerator=args.accelerator,
        logger=wandb_logger,
        deterministic=misc_args.force_deterministic,
        precision=args.precision,
    )
    trainer.test(model, datamodule=dm)
    # try to fix that test results sometimes do not get logged to wandb properly
    time.sleep(60)


if __name__ == "__main__":
    parsed_arg_groups = dargparse(dataclasses=(TrainingArgs, MiscArgs))
    main(parsed_arg_groups)
