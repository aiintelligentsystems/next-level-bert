from typing import TYPE_CHECKING
import nltk
import pandas as pd
import torch
import torchmetrics
from datasets import concatenate_datasets, load_dataset, Dataset, DatasetDict
import lightning as L
from loguru import logger
from torch.utils.data.dataloader import DataLoader
import datasets
import os
import re
import numpy as np
import sentence_transformers

from datasets import DatasetDict
from torchmetrics.retrieval import RetrievalMRR, RetrievalMAP, RetrievalHitRate
from torchmetrics import MetricCollection

if TYPE_CHECKING:
    from arguments import MiscArgs
    from arguments_evaluation import EvaluationArgs

from src.utils import spawn_processes_helper
from dlib.frameworks.pytorch import (
    set_torch_file_sharing_strategy_to_system,
)
from src.data_processing_library import (
    clean_str,
    create_batch_ids_downstream,
    create_input_chunks,
    create_input_sents,
    tokenize_dataset,
    sent_files_exist,
    merge_columns,
)

nltk.download("punkt")


class DownstreamDatamodule(L.LightningDataModule):
    def __init__(
        self,
        eval_args: "EvaluationArgs",
        misc_args: "MiscArgs",
        second_level_tokenizer,
        first_level_tokenizer,
        model,
    ):
        super().__init__()
        self.args = eval_args
        self.misc = misc_args
        self.num_preprocessing_splits = 1
        self.model = model
        self.recompute = self.args.recompute
        self.collate = model.downstream_collate
        if self.args.eval_dataset == "booksum_chapter":
            if (
                self.args.model_name_or_path != "PretrainedSBERT"
                and self.args.model_name_or_path != "PretrainedLongformer"
            ):
                self.dataset = DownstreamDatasetForNextLevel(
                    BookSumSTS(
                        first_level_tokenizer,
                        second_level_tokenizer,
                        self.args,
                        self.misc,
                    )
                )
            else:
                self.dataset = DownstreamDatasetForPretrainedBaseline(
                    BookSumSTS(
                        first_level_tokenizer,
                        second_level_tokenizer,
                        self.args,
                        self.misc,
                    )
                )
        elif self.args.eval_dataset == "ghomasHudson___muld":
            if (
                self.args.model_name_or_path != "PretrainedSBERT"
                and self.args.model_name_or_path != "PretrainedLongformer"
            ):
                self.dataset = DownstreamDatasetForNextLevel(
                    MuldMovieCharacters(
                        first_level_tokenizer,
                        second_level_tokenizer,
                        self.args,
                        self.misc,
                    )
                )
            else:
                self.dataset = DownstreamDatasetForPretrainedBaseline(
                    MuldMovieCharacters(
                        first_level_tokenizer,
                        second_level_tokenizer,
                        self.args,
                        self.misc,
                    )
                )
        elif self.args.eval_dataset == "quality":
            if (
                self.args.model_name_or_path != "PretrainedSBERT"
                and self.args.model_name_or_path != "PretrainedLongformer"
            ):
                self.dataset = DownstreamDatasetForNextLevel(
                    QualityQA(
                        first_level_tokenizer,
                        second_level_tokenizer,
                        self.args,
                        self.misc,
                    )
                )
            else:
                self.dataset = DownstreamDatasetForPretrainedBaseline(
                    QualityQA(
                        first_level_tokenizer,
                        second_level_tokenizer,
                        self.args,
                        self.misc,
                    )
                )
        else:
            raise Exception("Wrong downstream dataset name!")

    def prepare_data(self):
        self.dataset.download()
        if self.args.model_name_or_path in ["PretrainedSBERT", "PretrainedLongformer"]:
            splits = ["all"] if self.args.eval_dataset == "booksum_chapter" else ["train", "validation", "test"]
            split_file_paths = {
                split: self.dataset.processed_data_path + f"/{split}" + "/doc_level_dataset/"
                for split in splits
            }
            if not all([os.path.isdir(path) for path in split_file_paths.values()]) or self.recompute:
                logger.info("Preprocessing...")
                doc_parts = {}
                for split in splits:
                    self.dataset.prepare_data(split)
                    doc_parts[split] = datasets.Dataset.load_from_disk(
                    self.dataset.processed_data_path + f"/{split}/doc_level_dataset/"
                )
                doc_dataset = datasets.DatasetDict(doc_parts)
                doc_dataset.save_to_disk(self.dataset.processed_data_path + "/doc_level_datasets/")
                    
            else:
                logger.info(
                    "Only doc_level_dataset is needed. Aggregating splits, Skipping preprocessing..."
                )
                doc_parts = {
                    split: datasets.Dataset.load_from_disk(path)
                    for split, path in split_file_paths.items()
                }
                doc_dataset = datasets.DatasetDict(doc_parts)
                doc_dataset.save_to_disk(self.dataset.processed_data_path + "/doc_level_datasets/")
        elif (
            not self.recompute
            and sent_files_exist(self.dataset.processed_data_path)
            and os.path.isdir(self.dataset.processed_data_path + "/doc_level_datasets/")
            and os.path.isdir(self.dataset.processed_data_path + "/sent_datasets/")
        ):
            logger.info("Found processed data. Skipping preprocessing...")
        else:
            splits = (
                ["all"]
                if self.args.eval_dataset == "booksum_chapter"
                else ["train", "validation", "test"]
            )
            doc_parts = {}
            sent_data_parts = {}
            for split in splits:
                self.dataset.prepare_data(split)
            for split in splits:
                sent_data_parts[split] = datasets.Dataset.load_from_disk(
                    self.dataset.processed_data_path + f"/{split}/dataset_meta/"
                )
                doc_parts[split] = datasets.Dataset.load_from_disk(
                    self.dataset.processed_data_path + f"/{split}/doc_level_dataset/"
                )

            sent_data_splits = datasets.DatasetDict(sent_data_parts)
            sent_data_splits.save_to_disk(self.dataset.processed_data_path + "/sent_datasets/")
            doc_dataset = datasets.DatasetDict(doc_parts)
            doc_dataset.save_to_disk(self.dataset.processed_data_path + "/doc_level_datasets/")
        if self.args.data_preprocessing_only:
            exit(0)

    def setup(self, stage):
        if (
            self.args.model_name_or_path != "PretrainedSBERT"
            and self.args.model_name_or_path != "PretrainedLongformer"
        ):
            if (
                sent_files_exist(self.dataset.processed_data_path)
                and os.path.isdir(self.dataset.processed_data_path + "/doc_level_datasets/")
                and os.path.isdir(self.dataset.processed_data_path + "/sent_datasets/")
            ):
                splits = (
                    ["all"]
                    if self.args.eval_dataset == "booksum_chapter"
                    else ["train", "validation", "test"]
                )
                logger.info("Found processed data. Loading...")
                self.sentences_per_split = {}
                for split in splits:
                    if os.path.exists(self.dataset.processed_data_path + f"/{split}/sentences"):
                        self.sentences_per_split[split] = datasets.load_from_disk(
                            self.dataset.processed_data_path + f"/{split}/sentences"
                        )
                        self.downstream_dataset = datasets.DatasetDict.load_from_disk(
                            self.dataset.processed_data_path + "/doc_level_datasets/"
                        )
                        if split == "all":
                            self.sentences_per_split["test"] = self.sentences_per_split["all"]
                            self.downstream_dataset["test"] = self.downstream_dataset["all"]

                self.helper_data = datasets.DatasetDict.load_from_disk(
                    self.dataset.processed_data_path + "/sent_datasets/"
                )
                self.dataloader_data = create_batch_ids_downstream(
                    self.helper_data,
                    all_test=True if self.args.eval_dataset == "booksum_chapter" else False,
                )

            else:
                raise Exception(
                    f"Could not find processed data at {self.dataset.processed_data_path}. Please first prepare the data with --gpus 1 and --preprocessing_only first."
                )
        else:
            if os.path.isdir(self.dataset.processed_data_path + "/doc_level_datasets/"):
                self.dataloader_data = datasets.DatasetDict.load_from_disk(
                    self.dataset.processed_data_path + "/doc_level_datasets/"
                )
                self.downstream_dataset = self.dataloader_data
                if self.args.eval_dataset == "booksum_chapter":
                    # concatenate all splits into one test set
                    self.dataloader_data["test"] = self.dataloader_data["all"]
            else:
                raise Exception(
                    f"Could not find processed data at {self.dataset.processed_data_path}. Please first prepare the data with --gpus 1 and --preprocessing_only first."
                )

    def train_dataloader(self):
        if not self.dataset.needs_finetuning:
            return None
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=True,  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc.too_many_open_files_fix
            else None,
            shuffle=False,
            drop_last=True,
        )
        return DataLoader(self.dataloader_data["train"], collate_fn=self.collate, **common_args)

    def val_dataloader(self):
        if not self.dataset.needs_finetuning:
            return None
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=True,  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc.too_many_open_files_fix
            else None,
        )
        loader1 = DataLoader(
            self.dataloader_data["validation"], collate_fn=self.collate, **common_args
        )
        return loader1

    def test_dataloader(self):
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=True,  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc.too_many_open_files_fix
            else None,
        )
        loader1 = DataLoader(self.dataloader_data["test"], collate_fn=self.collate, **common_args)
        return loader1


class DownstreamDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        encoder,
        training_args: "EvaluationArgs",
        misc_args: "MiscArgs",
        path=None,
    ):
        super().__init__()
        self.args = training_args
        self.misc = misc_args
        self.encoder = encoder
        self.dataset_name = self.args.eval_dataset
        self.chunking = self.args.chunking
        self.processed_data_path = None
        self.num_preprocessing_splits = 1
        self.tokenizer = tokenizer
        self.num_proc = 8
        self.seed = 42
        self.sentence_counter = 0
        self.document_counter = 0
        self.properties = tuple()
        self.needs_finetuning = None
        self.output_size = None
        self.loss_fn = None
        self.default_save_path = (
            path or "/home/mamba/.cache/huggingface/datasets/" + self.args.eval_dataset
        )

    def preprocess(self, dataset):
        # clean text
        dataset_doc_level = dataset.map(clean_str, num_proc=self.num_proc, desc="Cleaning data...")
        dataset_doc_level = dataset_doc_level.filter(lambda x: len(x["text"]) > 0)
        return dataset_doc_level

    def download(self):
        raise NotImplementedError()

    def clean_dataset(self):
        raise NotImplementedError()

    def load(self, split):
        if not os.path.isdir(self.default_save_path + "/base"):
            self.download()
            self.clean_dataset()
        data = datasets.load_from_disk(self.default_save_path + "/base/")

        return data[split]

    def dataset_specific_preprocessing(self, data):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()


class DownstreamDatasetForNextLevel(torch.utils.data.Dataset):
    def __init__(self, downstream_dataset: DownstreamDataset):
        super().__init__()
        self.downstream_dataset = downstream_dataset
        self.processed_data_path = f"{downstream_dataset.default_save_path}/processed_{self.downstream_dataset.args.encoder_name}_{self.downstream_dataset.chunking}"
        self.needs_finetuning = self.downstream_dataset.needs_finetuning
        self.output_size = self.downstream_dataset.output_size
        self.num_proc = self.downstream_dataset.num_proc
        self.chunking = self.downstream_dataset.chunking
        self.encoder = self.downstream_dataset.encoder
        self.tokenizer = self.downstream_dataset.tokenizer
        self.args = self.downstream_dataset.args
        self.dataset_name = self.args.eval_dataset
        self.recompute = self.args.recompute

    def download(self):
        self.downstream_dataset.download()

    def compute(self, split="train"):
        if split == "all":
            data = []
            splits = ["train", "validation", "test"]
            for s in splits:
                data.append(self.downstream_dataset.load(s))
            data = concatenate_datasets(data)
        else:
            data = self.downstream_dataset.load(split)
        print(f"Processing split {split}.")
        data = self.downstream_dataset.dataset_specific_preprocessing(data)
        self.preprocess(data, split)

    def prepare_data(self, split="train"):
        self.sentence_counter = 0
        self.document_counter = 0

        if not os.path.isdir(
            self.processed_data_path
        ):
            os.makedirs(self.processed_data_path)
        path = self.processed_data_path + f"/{split}/"
        # check if file exists
        if os.path.isdir(path) and not self.recompute:
            logger.info(f"Found file at {path}.")
        elif (not os.path.isdir(path)) or self.recompute:
            logger.info(
                f"Could not find file at {path} or recompute has been set to True. Recomputing..."
            )
            self.compute(split)
        return

    def preprocess(self, dataset, split):
        """
        Preprocessing pipeline for the dataset. This includes:
        - cleaning the text
        - tokenizing the text
        - chunking the text
        - encoding the text with sbert multigpu support
        - creating start and end batch ids for sequences of chunks

        Returns: sentence_embeddings, dataset_doc_level, dataset_meta
        - sentence_embeddings: a huggingface dataset of shape (num_sentences + num_docs, encoder_embed_dim). After every document's sentences a zero vector is inserted to separate documents.
        - dataset_doc_level: dataset with documents per row. contains information for each document
        - dataset_meta: contains the tokenized text chunks before encoding
        """
        dataset_doc_level = self.downstream_dataset.preprocess(dataset)
        dataset_doc_level.save_to_disk(
            self.processed_data_path + f"/{split}/" + "doc_level_dataset/"
        )
        tokenized_data_path = f"{self.downstream_dataset.default_save_path}/tokenized_{split}_{self.args.encoder_name}"
        if os.path.isdir(tokenized_data_path) and not self.recompute:
            logger.info(f"Found tokenized data at {tokenized_data_path}. Loading...")
            dataset = Dataset.load_from_disk(tokenized_data_path)
        else:
            dataset = dataset_doc_level.map(
                lambda x: {
                    key: nltk.tokenize.sent_tokenize(value) if key == "text" else value
                    for key, value in x.items()
                },
                num_proc=self.num_proc,
            )
            del dataset_doc_level
            # filter out empty texts
            dataset = dataset.filter(lambda x: len(x["text"]) > 0)

            logger.info(
                f"Could not find tokenized data at {tokenized_data_path} or recompute has been set. Tokenizing..."
            )
            dataset = dataset.map(
                tokenize_dataset,
                num_proc=self.num_proc,
                desc="Tokenizing data...",
                fn_kwargs={"tokenizer": self.tokenizer, "column": "text"},
            )
            if self.args.eval_dataset == "ghomasHudson___muld":
                dataset = dataset.map(
                    tokenize_dataset,
                    num_proc=self.num_proc,
                    desc="Tokenizing extra chunk...",
                    fn_kwargs={"tokenizer": self.tokenizer, "column": "qa",},
                )
            if self.args.eval_dataset == "quality":
                dataset = dataset.map(
                    merge_columns,
                    num_proc=self.num_proc,
                    desc="Merging columns...",
                    fn_kwargs={"columns": ["question", "candidate"],
                               "new_column_name": "qa",},
                )
                dataset = dataset.map(
                    tokenize_dataset,
                    num_proc=self.num_proc,
                    desc="Tokenizing extra chunk...",
                    fn_kwargs={"tokenizer": self.tokenizer, "column": "qa",},
                )
            print(f"Trying to save file of size: {dataset.data.nbytes / 1e9} GB")
            dataset.save_to_disk(tokenized_data_path)

        # create chunks
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if self.args.eval_dataset == "quality" or self.args.eval_dataset == "ghomasHudson___muld":
            extra_chunk = True
        else:
            extra_chunk = False
        if self.args.chunking >= 1:
            chunk_func = create_input_chunks
        else:
            chunk_func = create_input_sents
        dataset = dataset.map(
            chunk_func,
            fn_kwargs={
                "cls_token_id": cls_token_id,
                "sep_token_id": sep_token_id,
                "pad_token_id": pad_token_id,
                "max_seq_len": self.encoder.max_seq_length,
                "chunking": self.chunking,
                "is_summarization": "is_summarization" in self.downstream_dataset.properties,
                "is_labeled": "is_labeled" in self.downstream_dataset.properties,
                "extra_chunk": extra_chunk,
            },
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            num_proc=self.num_proc,
            with_indices=True,
            desc="Creating chunks...",
        )
        dataset = dataset.add_column("sentence_ids", list(range(len(dataset))))
        # set columns to right format and compute embeddings in parallel
        # embeddings are saved to where save_file_path points to
        dataset.set_format(type="torch", columns=["chunks", "masks"], output_all_columns=True)
        dataset.save_to_disk(self.processed_data_path + f"/{split}/" + "dataset_meta/")
        dataset = dataset.load_from_disk(self.processed_data_path + f"/{split}/" + "dataset_meta/")
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
        logger.info(f"Saved processed dataset to {self.processed_data_path}/{split}/.")

    def evaluate(self, predictions, dataset):
        out = self.downstream_dataset.evaluate(predictions, dataset)
        return out


class DownstreamDatasetForPretrainedBaseline(torch.utils.data.Dataset):
    # class for baseline models that are not hierarchical and do not require text encoding during preprocessing
    def __init__(self, dataset: DownstreamDataset):
        super().__init__()
        self.downstream_dataset = dataset
        self.processed_data_path = f"/home/mamba/.cache/huggingface/datasets/{self.downstream_dataset.dataset_name}/processed_for_pretrained_baselines"
        self.needs_finetuning = self.downstream_dataset.needs_finetuning
        self.output_size = self.downstream_dataset.output_size
        self.recompute = self.downstream_dataset.args.recompute
        self.tokenizer = self.downstream_dataset.tokenizer
        self.num_proc = self.downstream_dataset.num_proc

    def download(self):
        self.downstream_dataset.download()

    def prepare_data(self, split="train"):
        if not os.path.isdir(
            self.processed_data_path
        ):
            os.makedirs(self.processed_data_path)
        path = self.processed_data_path + f"/{split}/"
        # check if file exists
        if os.path.isdir(path) and not self.recompute:
            logger.info(f"Found file at {path}.")
        elif (not os.path.isdir(path)) or self.recompute:
            logger.info(
                f"Could not find file at {path} or recompute has been set to True. Recomputing..."
            )
            self.compute(split)
        return

    def compute(self, split="train"):
        if split == "all":
            data = []
            splits = ["train", "validation", "test"]
            for s in splits:
                data.append(self.downstream_dataset.load(s))
            data = concatenate_datasets(data)
        else:
            data = self.downstream_dataset.load(split)
        print(f"Processing split {split}.")
        data = self.downstream_dataset.dataset_specific_preprocessing(data)
        self.preprocess(data, split)

    def preprocess(self, dataset, split):
        dataset_doc_level = self.downstream_dataset.preprocess(dataset)
        dataset_doc_level.save_to_disk(
            self.processed_data_path + f"/{split}/" + "doc_level_dataset/"
        )

    def evaluate(self, predictions, dataset):
        out = self.downstream_dataset.evaluate(predictions, dataset)
        return out


class MuldMovieCharacters(DownstreamDataset):
    def __init__(
        self,
        tokenizer,
        encoder,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
        path=None,
    ):
        super().__init__(tokenizer, encoder, training_args, misc_args, path)
        self.num_preprocessing_splits = 1
        self.properties = ("is_labeled",)
        self.needs_finetuning = True
        self.output_size = 1
        # this has effect on F1 score and loss weighting (for label imbalance)
        self.is_hero_pos_label = training_args.is_muld_pos_label_hero
        # pos_weight to account for label imbalance during training is manually calculated on train split
        # and hard coded
        if self.args.apply_class_balance_to_loss:
            if self.is_hero_pos_label:
                self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.37762]))
            else:
                self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.62238]))
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.available_splits = ("train", "validation", "test")

    def download(self):
        file_checks = [
            os.path.isfile(self.default_save_path + f"/base/{split}/data-00000-of-00001.arrow")
            for split in self.available_splits
        ]
        if not all(file_checks):
            dataset = load_dataset("ghomasHudson/muld", "Character Archetype Classification")
            logger.info(
                "Finished downloading 'ghomasHudson/muld Character Archetype Classification'"
            )
            dataset.save_to_disk(self.default_save_path + "/base")
            logger.info(
                f"Saved 'ghomasHudson/muld Character Archetype Classification' to {self.default_save_path + '/base'}"
            )
        else:
            logger.info("Found existing data files for all available splits, skip downloading.")

    # Needed because DownstreamDataset.load() is not overwritten
    # but it calls self.clean_dataset() which is not implemented in parent class
    def clean_dataset(self):
        pass

    def dataset_specific_preprocessing(
        self, data, **kwargs
    ):
        map_kwargs = kwargs
        if self.is_hero_pos_label:
            label2id = {"Villain/Antagonist": 0, "Hero": 1}
        else:
            label2id = {"Villain/Antagonist": 1, "Hero": 0}
        is_multi_split = isinstance(data, DatasetDict)

        def _create_document_title(example):
            # first line is character
            character_name = example["input"].partition("\n")[0].strip().lower()
            # movie title is usually in second line but inconsistent format
            # -> delete special characters / tags
            movie_title = example["input"].partition("\n")[2]  # everything after first row
            movie_title = movie_title.strip()  # remove trailing whitespaces
            movie_title = movie_title.partition("\n")[
                0
            ]  # first non-whitespace char sequence after first row
            movie_title = re.sub(r"<(\\)?[a-z]>", "", movie_title)  # remove HTML tags
            movie_title = re.sub('[<>/"]', "", movie_title)  # remove other distracting characters
            movie_title = movie_title.strip().lower()

            return {
                "document_title": movie_title + " (" + example["metadata"] + ") " + character_name,
                "qa": f"Is the character '{character_name}' behaving like a hero or a villain?",
            }

        def _flatten_check_single_label(example):
            if len(example["label_text"]) != 1:
                raise ValueError(
                    "Labels are mutually exclusive! Each example should have assigned exactly one label assigned"
                    f"but {len(example['label_text'])} were found."
                )
            return {"label_text": example["label_text"][0]}

        def _apply_transformations(data: Dataset):
            updated_data = data.map(
                _create_document_title,
                remove_columns=["metadata"],
                desc="Creating document titles...",
                **map_kwargs,
            )
            updated_data = updated_data.rename_column("input", "text")
            updated_data = updated_data.rename_column("output", "label_text")
            updated_data = updated_data.map(
                _flatten_check_single_label,
                desc="Check integrity and reformat labels...",
                **map_kwargs,
            )
            updated_data = updated_data.map(
                lambda x: {"label": label2id.get(x["label_text"], None)},
                desc="Transform labels to IDs...",
            )
            return updated_data

        # apply transformations to all splits if a whole DatasetDict is provided
        if is_multi_split:
            updated_data = DatasetDict()
            for split in data.keys():
                updated_data[split] = _apply_transformations(data[split])
        # or a single one otherwise
        else:
            updated_data = _apply_transformations(data)

        return updated_data

    def evaluate(self, predictions, dataset, return_binary=False):
        predictions = predictions.cpu().squeeze()
        targets = torch.LongTensor(dataset["label"]).cpu().squeeze()
        correct = torch.sigmoid(predictions).round().int() == targets
        acc_manual = correct.float().mean()
        metric = torchmetrics.classification.BinaryAccuracy()
        # assumes that predictions are ordered by document id of split
        accuracy = metric(torch.sigmoid(predictions), targets.int().cpu())
        print(acc_manual)
        metric2 = torchmetrics.classification.BinaryF1Score()
        f1 = metric2(torch.abs(torch.sigmoid(predictions) - 1), torch.abs(targets.int().cpu() - 1))
        if return_binary:
            return {"accuracy":accuracy.item(), "f1":f1.item(), 'is_correct':correct}
        return {"accuracy":accuracy.item(), "f1":f1.item()}


class BookSumSTS(DownstreamDataset):
    def __init__(
        self,
        tokenizer,
        encoder,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
    ):
        super().__init__(tokenizer, encoder, training_args, misc_args)
        self.num_preprocessing_splits = 1
        self.properties = "is_summarization"
        self.needs_finetuning = False

    def download(self):
        path = "/home/mamba/.cache/huggingface/datasets/booksum"
        if not os.path.isdir(path):
            load_dataset("kmfoda/booksum")

    def clean_dataset(self):
        raise NotImplementedError()

    def load(self, split):
        path = "/home/mamba/.cache/huggingface/datasets/booksum"
        if not os.path.isdir(path):
            self.download()
        data = datasets.load_from_disk(path)
        return data[split]

    def dataset_specific_preprocessing(self, data):
        # append column 'summary' to 'text'. add marker to title column

        data_pd = data.to_pandas()
        data_pd["summaries"] = data_pd["summaries"].apply(tuple)
        data_pd = data_pd.drop_duplicates(ignore_index=True, keep="first")
        data = Dataset.from_pandas(data_pd)

        def remove_duplicate_summaries(examples):
            summaries = examples["summaries"]
            deduplicated_summaries = []
            for doc in summaries:
                unique_summaries = list(set(doc))
                deduplicated_summaries.append(unique_summaries)
            return {
                "text": examples["text"],
                "summaries": deduplicated_summaries,
                "title": examples["title"],
            }

        dataset = data.map(
            remove_duplicate_summaries,
            batched=True,
            batch_size=100000,
            remove_columns=["summaries"],
            desc="Removing duplicate summaries...",
        )

        def append_summary(examples):
            flattened_summaries = []
            flattened_summary_titles = []
            summary_counter = 0
            is_summary = [False] * len(examples["text"])
            summary_id = [-1] * len(examples["text"])
            titles = examples["title"]
            for i, summary in enumerate(examples["summaries"]):
                for j, sub_summary in enumerate(summary):
                    flattened_summaries.append(sub_summary)
                    flattened_summary_titles.append(titles[i])
                    is_summary.append(True)
                    summary_id.append(summary_counter)
                    summary_counter += 1
            texts = examples["text"] + flattened_summaries
            titles = examples["title"] + flattened_summary_titles
            return {
                "text": texts,
                "document_title": titles,
                "is_summary": is_summary,
                "summary_id": summary_id,
            }

        dataset = dataset.map(
            append_summary,
            batched=True,
            batch_size=100000,
            remove_columns=["summaries", "title", "text"],
            desc="Appending summaries...",
        )
        return dataset

    def evaluate(self, predictions, dataset):
        import faiss

        # for each full text chapter, look within the set of summaries for the ones with the highest cosine similarity. (some chapters have multiple summaries)
        vectors = np.array(predictions.cpu(), dtype=np.float32)
        faiss.normalize_L2(vectors)
        dataset = dataset.add_column("norm_vecs", list(vectors))
        self.metrics = MetricCollection(
            [RetrievalMRR(), RetrievalMAP(), RetrievalHitRate()],
        )
        summaries = dataset.filter(lambda x: x["is_summary"] is True)
        full_texts = dataset.filter(lambda x: x["is_summary"] is False)
        queries = np.array(full_texts["norm_vecs"], dtype=np.float32)
        summaries = summaries.add_faiss_index(column="norm_vecs")
        print("Get nearest neighbor example")

        scores, retrieved_examples = summaries.get_nearest_examples_batch(
            "norm_vecs", queries, k=10
        )
        gt = self.get_gt(dataset)
        gt = self.get_text_lengths(gt)
        retrieved_sums = [ex["summary_id"] for ex in retrieved_examples]
        gt["retrieved_examples"] = retrieved_sums
        gt["scores"] = scores
        gt.to_pickle(f"output/{self.dataset_name}/{self.args.model_name_or_path}_{self.args.encoder_name}.csv")
        preds, targs, indeces = self.prepare_metric_format(gt)
        metrics = self.metrics(preds, targs, indeces)
        return metrics

    def prepare_metric_format(self, gt):
        scores = gt["scores"].tolist()
        scores = np.ones_like(scores) - scores
        targets = []
        for i, row in gt.iterrows():
            target = [id_num in row["summary_id"] for id_num in row["retrieved_examples"]]
            targets.append(target)
        targets = torch.tensor(targets)
        indeces = ((torch.arange(len(scores))).unsqueeze(1)).repeat(1, len(scores[0]))
        predictions = torch.tensor(scores)
        return predictions, targets, indeces

    def get_gt(self, dataset):
        df = dataset.to_pandas()
        full_texts = df[df["is_summary"] == False]
        summaries = df[df["is_summary"] == True]
        tmp = (
            summaries.groupby("document_title")["summary_id"]
            .apply(list)
            .reset_index(name="summary_id")
        )
        tmp = tmp.sort_values("summary_id", ignore_index=True)
        tmp = full_texts.merge(tmp, "outer", on="document_title")
        # remove duplicate column content
        tmp = tmp.drop(columns=["summary_id_x"])
        tmp = tmp.rename(columns={"summary_id_y": "summary_id"})
        tmp = tmp.sort_values("summary_id", ignore_index=True)
        return tmp

    def get_text_lengths(self, df):
        tokenizer = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2").tokenizer
        df["tokenized"] = df["text"].apply(
            lambda example: tokenizer(example, truncation=False, padding=False)["input_ids"],
        )
        df["len"] = df["tokenized"].apply(lambda x: len(x))
        return df


class QualityQA(DownstreamDataset):
    def __init__(
        self,
        tokenizer,
        encoder,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
        path=None,
    ):
        super().__init__(tokenizer, encoder, training_args, misc_args, path)
        self.num_preprocessing_splits = 1
        self.properties = ("is_labeled",)
        self.needs_finetuning = True
        self.output_size = 1
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.available_splits = ("train", "validation", "test")

    def download(self):
        file_checks = [
            os.path.isfile(self.default_save_path + f"/raw/QuALITY.v1.0.1.htmlstripped.{split}")
            if split != "validation"
            else os.path.isfile(self.default_save_path + "/raw/QuALITY.v1.0.1.htmlstripped.dev")
            for split in self.available_splits
        ]
        if not all(file_checks):
            logger.error(
                "Download Quality with external bash script in scripts/download_quality.sh."
            )

        else:
            logger.info("Found existing data files for all available splits, skip downloading.")

    # Needed because DownstreamDataset.load() is not overwritten
    # but it calls self.clean_dataset() which is not implemented in parent class
    def clean_dataset(self):
        data = {}
        file = [file for file in os.listdir(self.default_save_path + "/raw/") if "train" in file]
        df = pd.read_json(self.default_save_path + "/raw/" + file[0], lines=True)
        data["train"] = Dataset.from_pandas(df)
        # use dev file as test file since for the real test split labels are not provided yet
        val_file = [file for file in os.listdir(self.default_save_path + "/raw/") if "dev" in file]
        val_data = pd.read_json(self.default_save_path + "/raw/" + val_file[0], lines=True)
        data["test"] = Dataset.from_pandas(val_data)
        # create test set from dev set (given test set is without labels for leaderboard)
        split_data = datasets.Dataset.train_test_split(data["test"], test_size=0.5, seed=42)
        data["validation"] = split_data["train"]
        data["test"] = split_data["test"]
        data = datasets.DatasetDict(data)
        data.save_to_disk(self.default_save_path + "/base/")

    def dataset_specific_preprocessing(self, data, **kwargs):
        texts = []
        questions = []
        candidates = []
        labels = []
        document_titles = []
        unique_ids = []

        def create_examples(examples):
            for i, article in enumerate(examples["article"]):
                for j, question in enumerate(examples["questions"][i]):
                    for k, option in enumerate(question["options"]):
                        text = article #+ ". " + question["question"] + ". " + option
                        texts.append(text)
                        labels.append(question["gold_label"])
                        questions.append(question["question"])
                        candidates.append(option)
                        document_titles.append(examples["title"][i])
                        unique_ids.append(question["question_unique_id"])
            return {
                "text": texts,
                "question": questions,
                "candidate": candidates,
                "label": labels,
                "document_title": document_titles,
                "unique_id": unique_ids,
            }

        data = data.map(
            create_examples,
            batched=True,
            batch_size=100,
            num_proc=self.num_proc,
            remove_columns=data.column_names,
            desc="Creating individual examples...",
        )
        return data
    
    def evaluate(self, predictions, dataset):
        predictions = predictions.cpu().squeeze()
        targets = torch.LongTensor(dataset["label"]).cpu().squeeze().view(-1, 4)[:,0] - 1
        metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=4)
        accuracy = metric(torch.softmax(predictions, -1), targets)
        print(accuracy)
        return accuracy
