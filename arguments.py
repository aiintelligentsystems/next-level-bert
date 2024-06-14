from typing import Literal
from dataclasses import dataclass
from dargparser import dArg


@dataclass
class TrainingArgs:
    model_name_or_path: str = dArg(
        default="NextLevelLM",
        help="HuggingFace model identifier. This is used to construct the model architecture and load pretrained weights if not specified otherwise.",  # noqa: E501
        aliases="--model",
    )
    resume_training: bool = dArg(
        default=False,
        help="Whether to resume training form checkpoint or only load the weights.",
        aliases="--resume",
    )
    checkpoint_path: str | None = dArg(
        default=None,
        help="Path to a saved pytorch-lightning checkpoint. Use the wandb:<wandb-run-id> syntax to load a checkpoint from W&B.",  # noqa: E501
        aliases="--checkpoint",
    )

    dataset: Literal["pile_books", "pile"] = dArg(
        default="pile_books",
        help="Dataset to use for pretraining. Currently only pile_books supported.",
    )
    max_sequence_length: int = dArg(
        default=512,
        help="Sequence length of the next-level model.",
        aliases=["--max_seq_length", "--block_size"],
    )
    overwrite_data_cache: bool = dArg(
        default=False, help="Overwrite the cached preprocessed datasets or not.", aliases="--odc"
    )
    conserve_disk_space: bool = dArg(
        default=False, help="Cleanup cache files whenever possible to save disk space."
    )
    data_preprocessing_only: bool = dArg(
        default=False, help="Exit the script after data preprocessing. Do not start training."
    )
    only_eval: bool = dArg(
        default=False,
        help="Whether to only do evaluation on a downstream task. If True, must also specify --checkpoint_path and --eval_dataset",
    )
    eval_dataset: Literal["ghomasHudson___muld", "booksum_chapter", "quality"] = dArg(
        default="booksum_chapter",
        help="Dataset to use for evaluation. Can be 'ghomasHudson___muld', 'booksum_chapter', 'quality'.",
    )
    encoder_name: str = dArg(
        default="all-MiniLM-L6-v2",
        help="The exact model from the sentence transformers/SBERT library used in either the SBERT baseline or as the second-level tokenizer.",
    )
    apply_class_balance_to_loss: bool = dArg(
        default=False,
        help=("Whether or not to apply a (pre-computed) loss weight for different classes to counter class imbalance. "
              "If the DownstramDataset does not implement this option the argument has no effect"
              ),
    )
    chunking: int | None = dArg(
        default=None,
        help="Which chunking strategy to use for encoding the input with the second-level tokenizer",
    )
    second_level_tokenizer_batch_size: int = dArg(
        default=2048,
        help="What batch size to use for computing the sentence vectors during preprocessing.",
    )
    loss_func: str = dArg(
        default="smoothl1", help="Which loss function to use for pretraining. Choose from cosine, l1 or l2."
    )
    
    ####### Hardware ###########
    accelerator: Literal["gpu", "cpu", "tpu", "mps", "auto"] = dArg(
        default="auto",
        help='Hardware accelerator to use. Can be gpu, cpu, tpu, mps, etc. If "auto", will auto-detect available hardware accelerator.',  # noqa: E501
    )
    distributed_strategy: Literal["ddp", "ddp_smart", "ddp_spawn", "ddp_fork", "dp", "auto"] = dArg(
        default="auto",
        help="Distributed training strategy to use.",
        aliases=["--dist_strategy", "--ds"],
    )
    devices: int | None = dArg(
        default=None,
        aliases=["--gpus", "--cpus", "--tpus"],
        help="Number of devices to use for distributed training. If -1, will use all available devices.",  # noqa: E501
    )
    workers: int = dArg(
        default=8,
        help="Number of workers for dataloaders. *Every device* weill use that many workers.",
        aliases="-w",
    )
    preprocessing_workers: int = dArg(
        default=4,
        help="Number of workers for preprocessing the datasets. Cached datasets are only valid for the same number of preprocessing workers.",  # noqa: E501
        aliases="--pw",
    )
    precision: Literal["16-mixed", "bf16-mixed", 32] = dArg(
        default=32,
        help="Floating point precision to use during training. Might require specific hardware.",
    )
    compile: bool = dArg(
        default=False,
        help="Whether to compile the model with using `torch.compile`. Requires torch>=2.0",
    )
    debug: bool = dArg(
        default=False,
        help="If true, wait for debugger to attach at the start of the script.",
    )
    recompute: bool = dArg(
        default=False,
        help="Whether to completely recompute all data preprocessing without cached datasets.",
    )
    ####### General training ###########
    max_epochs: int = dArg(default=50, help="Max number of samples")
    val_frequency: float = dArg(
        default=200,
        help="Do validation every K samples.",
        aliases="--vfq",
    )
    val_batches: float | None = dArg(
        default=None,
        help="Limit validation set to k batches.",
    )
    model_log_frequency: int = dArg(
        default=500,
        help="Log a model checkpoint every K samples.",
        aliases="--mfq",
    )
    val_before_training: bool = dArg(
        default=False, help="Run one validation epoch before training."
    )
    batch_size_per_device: int = dArg(
        default=8,
        help="Batch size per device. If --effective_batch_size is specified, this is the maximum batch size per device (you should test when you cannot get larger without CUDA OOM errors.).",  # noqa: E501
        aliases=["--batch_size_per_gpu", "-b"],
    )
    effective_batch_size: int | None = dArg(
        default=None,
        help="If set, try to auto-infer batch_size_per_device and gradient_accumulation_steps based on number of devices given by --devices.",  # noqa: E501
        aliases=["--eb"],
    )
    only_tune_cls_head: bool = dArg(
        default=False,
        help="If True, during fine-tuning the optimizer only receives the parameters of the cls head and the rest of the network is frozen.",
        aliases=["--cls_only"],
    )
    learning_rate: float = dArg(default=5e-5, aliases="--lr")
    lr_warmup: float = dArg(
        default=0.05,
        help="Number of K samples to do a learning rate warmup. If <1, compute as fraction of training_goal.",  # noqa: E501
    )
    weight_decay: float = dArg(default=0.0, aliases="--wd")
    gradient_clipping: float | None = dArg(default=None, aliases="--gc")
    gradient_accumulation_steps: int = dArg(default=1, aliases=["--gas", "--accum"])
    early_stopping: bool = dArg(default=False, help="Whether to use early stopping or not.")
    early_stopping_patience: int = dArg(
        default=5, help="Number of epochs to wait before early stopping."
    )
    early_stopping_metric: str = dArg(
        default="val/loss", help="Metric to use for early stopping."
    )

    ### Dataset specific flags
    is_muld_pos_label_hero: bool = dArg(
        default=False,
        help=("Whether hero or villain is the positive lablel which makes a difference e.g. for F1-score. "
              "Hero is the majority class, so by default it is defined as negative label 0.")
    )

    ### Model
    dropout: float = dArg(default=0.1, help="Dropout for custom NextLevelLM. Ignored otherwise.")
    nhead: int = dArg(default=4, help="Number of Heads for custom NextLevelLM. Ignored otherwise.")
    num_layers: int = dArg(default=2, help="Number of Layers for custom NextLevelLM. Ignored otherwise.")

    masking_probability: float = dArg(default=0.15, help="Masking probability")
    evaluate_downstream: bool = dArg(
        default=False, help="Whether to evaluate the model on a downstream task."
    )
    mask_scheme: Literal["original_bert", "only_mask"] = dArg(
        default="original_bert", help="Masking scheme to use."
    )
    aggregate: Literal["all", "last", "concat", "label", "cls", "pool"] = dArg(
        default="concat",
        help=("Selects a different representation as input to the DownstreamModel's classifier head. Has no effect for other models."
              "('all' = document vector + SBERT extra chunk, 'last' = extra chunk only, 'concat' = concattenation of document vector + extra chunk (next level encoded),"
              " 'cls' = only CLS token embedding, 'pool' = document vector only, 'label' = test label info leak debugging only)"
              ),
    )
    pooling_aggregator: Literal["mean", "max"] = dArg(
        default="mean",
        help=("Type of pooling to apply to create a document embedding from the sequence. "
              "'mean' (default) = average pooling, 'max' = max pooling"),
    )
    use_custom_roberta_config: bool = dArg(
        default=False,
        help="Whether to use a customizable RoBERTa-based next-level model, instead of initializing the next-level model with the"
        "encoder model's architecture and weights. This model will have randomly initialized weights and can be customized with the flags --dropout --num_layers --num_heads.",
    )

    def __post_init__(self):
        if self.distributed_strategy is None:
            self.devices = None
        elif self.devices is None:
            self.distributed_strategy = "auto"


@dataclass
class MiscArgs:
    seed: int | None = dArg(default=42, help="Random seed to use.")
    force_deterministic: bool = dArg(
        default=False, aliases="-fd", help="Force PyTorch operations to be deterministic."
    )
    offline: bool = dArg(default=False, help="Disable W&B online syncing.")
    fast_dev_run: bool = dArg(
        default=False, help="Do fast run through training and validation with reduced sizes."
    )
    wandb_run_name: str | None = dArg(
        default=None, help="Run name for the W&B online UI.", aliases="-n"
    )
    wandb_tags: list[str] = dArg(default=[])
    wandb_project: str = dArg(default=None)
    too_many_open_files_fix: bool = dArg(
        default=False,
        help='Apply fix to circumvent "Too many open files" error caused by the PyTorch Dataloader when using many workers or large batches.',  # noqa: E501
        aliases="--open_files_fix",
    )
    output_dir: str = dArg(
        default="output",
        help="Output directory for downstream predictions",
        aliases="--od",
    )
