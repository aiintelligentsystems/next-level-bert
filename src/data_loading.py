from typing import TYPE_CHECKING
import nltk
import torch
import numpy as np
from datasets import load_dataset, Dataset
import lightning as L
from loguru import logger
from torch.utils.data.dataloader import DataLoader
import datasets
import os
import warnings
from datasets import DatasetDict
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from train import MiscArgs, TrainingArgs

from dlib.frameworks.pytorch import (
    set_torch_file_sharing_strategy_to_system,
)
from datasets import disable_caching
from src.utils import spawn_processes_helper
from src.data_processing_library import (
    clean_str,
    delete_dataset,
    tokenize_dataset,
    create_input_chunks,
    create_input_sents,
    sent_files_exist,
)
from src.customize_dataset import _iter_pytorch

nltk.download("punkt")
disable_caching()
warnings.filterwarnings("ignore", message="^.*promote has been superseded.*$")
warnings.filterwarnings(
    "ignore", message="^.*Running this sequence through the model will result in indexing errors.*$"
)
# necessary since the original implementation of the dataloader iter function in combination with huggingface datasets does not retrieve contiguous data from memory
setattr(datasets.IterableDataset, '_iter_pytorch', _iter_pytorch)

def _get_second_level_tokenizer_dim(second_level_tokenizer: SentenceTransformer):
    return next(second_level_tokenizer.parameters()).shape[1]


class NextLevelDatamodule(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
        second_level_tokenizer,
        first_level_tokenizer,
        model,
    ):
        super().__init__()
        self.args = training_args
        self.misc = misc_args
        self.subset_size = None
        self.model = model
        self.collate = lambda x: model.pretraining_collate(
            x, masking_rate=self.args.masking_probability
        )
        self.embed_dim = _get_second_level_tokenizer_dim(second_level_tokenizer)
        self.drop_last_batch = True
        if self.args.dataset == "pile_books":
            self.dataset = PileBooks(
                first_level_tokenizer,
                second_level_tokenizer,
                self.args,
                self.misc,
                len_subset=self.subset_size,
                recompute=self.args.recompute,
            )
        else:
            raise Exception("Wrong dataset name!")

    def prepare_data(self):
        #############################################################################
        ####################### Preprocess pretraining dataset ######################
        #############################################################################
        self.dataset.download()
        if not self.args.recompute and sent_files_exist(self.dataset.processed_data_path):
            logger.info("Found processed data. Skipping preprocessing...")
        else:
            splits = ["train", "validation", "test"]
            for split in splits:
                self.dataset.prepare_data(split)
        if self.args.data_preprocessing_only:
            exit(0)

    def setup(self, stage):
        #############################################################################
        ####################### Load pretraining dataset ######################
        #############################################################################

        if sent_files_exist(self.dataset.processed_data_path):
            splits = ["train", "validation", "test"]
            self.sentences_per_split = {}
            logger.info(f"Found processed data at {self.dataset.processed_data_path} Loading...")
            data_files_dict = {}
            for split in splits:
                if os.path.exists(self.dataset.processed_data_path + f"/{split}/sentences"):
                    logger.info("loading from datasets file")
                    # gather all data files in a dict if file ends in .arrow
                    data_files_dict[split] = [
                        os.path.join(self.dataset.processed_data_path + f"/{split}/sentences", file)
                        for file in os.listdir(
                            self.dataset.processed_data_path + f"/{split}/sentences"
                        )
                        if file.endswith(".arrow")
                    ]
                else:
                    raise Exception(
                        f"Could not find processed data at {self.dataset.processed_data_path}/{split}/sentences. Please first prepare the data with --gpus 1 and --preprocessing_only."
                    )
            self.sentences_per_split = load_dataset(
                "arrow", data_files=data_files_dict, streaming=True
            ).with_format("torch")
            for split in splits:
                self.sentences_per_split[split].batch_size = self.args.batch_size_per_device * (self.args.max_sequence_length - 1)
                self.sentences_per_split[split].drop_last_batch = self.drop_last_batch

        logger.info("Loaded dataset...")

    def train_dataloader(self):
        common_args = dict(
            batch_size=None,
            num_workers=self.args.workers,
            persistent_workers=True,  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc.too_many_open_files_fix
            else None,
            shuffle=False,
        )
        return DataLoader(self.sentences_per_split["train"], collate_fn=self.collate, **common_args)

    def val_dataloader(self):
        common_args = dict(
            batch_size=None,
            num_workers=self.args.workers,
            persistent_workers=True,  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc.too_many_open_files_fix
            else None,
        )
        loader1 = DataLoader(
            self.sentences_per_split["validation"], collate_fn=self.collate, **common_args
        )
        return loader1

    def test_dataloader(self):
        common_args = dict(
            batch_size=None,
            num_workers=self.args.workers,
            persistent_workers=True,  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc.too_many_open_files_fix
            else None,
        )
        loader1 = DataLoader(
            self.sentences_per_split["test"], collate_fn=self.collate, **common_args
        )
        return loader1


class PretrainingDataset(object):
    """
    Class for all datasets used for pretraining. It implements the following methods:
    - download: downloads the dataset
    - preprocess: preprocesses the dataset
    - prepare_data: returns the preprocessed dataset
    - load_data: loads the downloaded dataset from disk
    - dataset_specific_preprocessing: dataset specific preprocessing to be implemented in subclasses
    - compute_from_part_i: computes the dataset from part i to num_preprocessing_splits. Is used to pick up where preprocessing left off
    """

    def __init__(
        self,
        tokenizer,
        encoder,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
        len_subset=None,
        recompute=False,
    ):
        super().__init__()
        self.args = training_args
        self.misc = misc_args
        self.encoder = encoder
        self.dataset_name = self.args.dataset
        self.len_subset = len_subset
        self.chunking = self.args.chunking
        self.num_preprocessing_splits = 5
        self.processed_data_path = f"/home/mamba/.cache/huggingface/datasets/{self.dataset_name}/processed_{self.args.encoder_name}_{self.chunking}_{self.len_subset}"
        self.tokenizer = tokenizer
        self.num_proc = 24
        self.seed = 42
        self.recompute = recompute
        self.sentence_counter = 0
        self.document_counter = 0

    def __len__(self):
        return len(self.data)

    def download(self):
        raise NotImplementedError()

    def compute_data(self, split="train"):
        data = self.load_data(split)
        print(f"Processing split {split}.")
        data = self.dataset_specific_preprocessing(data)
        self.preprocess(data, split)

    def prepare_data(self, split="train"):
        self.sentence_counter = 0
        self.document_counter = 0

        if not os.path.isdir(
            self.processed_data_path
        ):
            os.makedirs(self.processed_data_path)
        path = self.processed_data_path + f"/{split}/sentences/"
        # check if file exists
        if os.path.isdir(path) and not self.recompute:
            logger.info(f"Found file at {path}.")
        elif (not os.path.isdir(path)) or self.recompute:
            logger.info(
                f"Could not find file at {path} or recompute has been set to True. Recomputing..."
            )
            self.compute_data(split)
        return

    def preprocess(self, dataset, split):
        """
        Preprocessing pipeline for the dataset. This includes:
        - cleaning the text
        - tokenizing the text
        - chunking the text
        - encoding the text with sbert multigpu support
        - creating start and end batch ids for sequences of chunks

        saves a chunk embedding file
        - sentence embeddings: memory-mapped dataset with torch format of shape (num_sentences + num_docs, encoder_embed_dim). After every document's sentences a zero vector is inserted to separate documents.
        """
        # prepare directory for encoded data
        if not os.path.exists(self.processed_data_path + f"/{split}/"):
            os.makedirs(self.processed_data_path + f"/{split}/")
        tokenized_data_path = f"/home/mamba/.cache/huggingface/datasets/{self.dataset_name}/tokenized_{split}_{self.args.encoder_name}_"
        if os.path.isdir(tokenized_data_path + "None") and not self.recompute:
            logger.info(f"Found tokenized data at {tokenized_data_path}None. Loading...")
            dataset = Dataset.load_from_disk(tokenized_data_path + "None")
            if self.len_subset is not None:
                dataset = dataset.select(range(self.len_subset))
        elif os.path.isdir(tokenized_data_path + f"{self.len_subset}") and not self.recompute:
            logger.info(
                f"Found tokenized data at {tokenized_data_path}{self.len_subset}. Loading..."
            )
            dataset = Dataset.load_from_disk(tokenized_data_path + f"{self.len_subset}")
        else:
            # clean text
            # This line creates a new dataset using map, and deletes the old dataset
            dataset, _ = dataset.map(clean_str, num_proc=self.num_proc), delete_dataset(dataset)
            # create sentence chunks
            dataset = dataset.filter(lambda x: len(x["text"]) > 0)
            dataset, _ = dataset.map(
                lambda x: {
                    "text": nltk.tokenize.sent_tokenize(x["text"]),
                    "document_title": x["document_title"],
                },
                num_proc=self.num_proc,
            ), delete_dataset(dataset)
            # filter out empty texts
            dataset = dataset.filter(lambda x: len(x["text"]) > 0)

            logger.info(
                f"Could not find tokenized data at {tokenized_data_path} or recompute has been set. Tokenizing..."
            )
            dataset, _ = dataset.map(
                tokenize_dataset,
                remove_columns=["text"],
                num_proc=self.num_proc,
                desc="Tokenizing data...",
                fn_kwargs={"tokenizer": self.tokenizer},
            ), delete_dataset(dataset)
            print(f"Trying to save file of size: {dataset.data.nbytes / 1e9} GB")
            dataset.save_to_disk(tokenized_data_path + f"{self.len_subset}")

        # create chunks
        chunks_data_path = f"/home/mamba/.cache/huggingface/datasets/{self.dataset_name}/chunks_{split}_{self.args.encoder_name}_{self.chunking}"
        if os.path.isdir(chunks_data_path) and not self.recompute:
            logger.info(f"Found chunked data at {chunks_data_path}. Loading...")
            dataset = Dataset.load_from_disk(chunks_data_path)
        else:
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            pad_token_id = self.tokenizer.pad_token_id
            if self.args.chunking >= 1:
                chunk_func = create_input_chunks
            else:
                chunk_func = create_input_sents
            dataset, _ = dataset.map(
                chunk_func,
                fn_kwargs={
                    "cls_token_id": cls_token_id,
                    "sep_token_id": sep_token_id,
                    "pad_token_id": pad_token_id,
                    "max_seq_len": self.encoder.max_seq_length,
                    "chunking": self.chunking,
                    "is_summarization": False,
                },
                batched=True,
                batch_size=32,
                remove_columns=dataset.column_names,
                num_proc=self.num_proc,
                with_indices=True,
                desc="Creating chunks...",
                writer_batch_size=500,
            ), delete_dataset(dataset)
            dataset.save_to_disk(chunks_data_path)

        # set columns to right format and compute embeddings in parallel
        # embeddings are saved to where save_file_path points to
        # encodes text chunks with all available GPUs
        dataset.set_format(type="torch", columns=["chunks", "masks"], output_all_columns=True)
        spawn_processes_helper(
            dataset,
            fn_kwargs={
                "encoder_name": self.args.encoder_name,
                "batch_size": self.args.encoder_batch_size,
            },
            remove_columns=["chunks", "masks"],
            desc="Encoding chunks...",
            batched=True,
            batch_size=int(self.args.encoder_batch_size),
            save_file_path=self.processed_data_path + f"/{split}/",
        )
        del dataset
        logger.info(f"Saved processed dataset(s) to {self.processed_data_path}/{split}/.")

    def dataset_specific_preprocessing(self):
        raise NotImplementedError()

    def load_data(self):
        raise NotImplementedError()


class PileBooks(PretrainingDataset):
    def __init__(
        self,
        tokenizer,
        encoder,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
        len_subset=None,
        recompute=False,
    ):
        super().__init__(tokenizer, encoder, training_args, misc_args, len_subset, recompute)
        self.split_data_path = "/home/mamba/.cache/huggingface/datasets/pile_books/data_splits"

    def download(self):
        if os.path.isdir(self.split_data_path):
            return
        else:
            data = load_dataset(
                "arrow",
                data_dir="/home/mamba/.cache/huggingface/datasets/the_pile_books3/plain_text/1.0.0/b117651725f2603975a3e5ee0264081f0a448b5b4818bf5d1a4aabcf4416bc23/",
            )["train"]
            split1 = data.train_test_split(test_size=0.2, seed=self.seed)
            split2 = split1["test"].train_test_split(test_size=0.5, seed=self.seed)
            dataset = {
                "train": split1["train"].flatten_indices(),
                "validation": split2["train"].flatten_indices(),
                "test": split2["test"].flatten_indices(),
            }
            dataset = DatasetDict(dataset)
            dataset = dataset
            dataset.save_to_disk(self.split_data_path)

    def dataset_specific_preprocessing(self, dataset):
        dataset = dataset.rename_column("title", "document_title")
        return dataset

    def load_data(self, split):
        if os.path.isdir(self.split_data_path):
            data = datasets.load_from_disk(self.split_data_path)
            if self.len_subset is not None:
                data["train"] = data["train"].select(
                    range(min(int(self.len_subset * 0.8), len(data["train"])))
                )
                data["validation"] = data["validation"].select(
                    range(min(int(self.len_subset * 0.1), len(data["validation"])))
                )
                data["test"] = data["test"].select(
                    range(min(int(self.len_subset * 0.1), len(data["test"])))
                )
        else:
            logger.error("Could not find split data. Please run download first.")
        return data[split]
