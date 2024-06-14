from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Union, List, Dict

import lightning as L
import torch
from loguru import logger
from torch.optim import AdamW
from torch import nn
import numpy as np
import math
import queue
import os
import re
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import sentence_transformers
import torch.multiprocessing as mp


from transformers import RobertaConfig, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from torchmetrics.classification import BinaryAccuracy, Accuracy

# from src.evaluate import DocumentSimilarity
from tqdm.autonotebook import trange
import wandb

from src.roberta_wrapper import CustomRoberta, CustomRobertaEmbeddings

# from transformers.optimization import get_scheduler
# from warmup_scheduler import GradualWarmupScheduler

# from dlib.frameworks.pytorch import get_rank

if TYPE_CHECKING:
    from train import TrainingArgs


@dataclass
class ModelArgs:
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


class SimpleClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, 384), nn.GELU(), nn.Linear(384, config.hidden_size)
        )

    def forward(self, sequence_output):
        return self.cls(sequence_output)


def _extract_trainer_stage(run_stage_object):
    if run_stage_object == "train":
        return "train"
    if run_stage_object == "validate":
        return "validation"
    if run_stage_object == "test":
        return "test"
    return


class NextLevelLM(L.LightningModule):
    def __init__(
        self,
        training_args: "TrainingArgs",  # do in string to remove dependency when loading.
        adhoc_args: ModelArgs = ModelArgs(),
        second_level_tokenizer: SentenceTransformer = None,
        dropout: float = 0.1,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if not training_args.resume_training:
            self.save_hyperparameters(
                ignore=["effective_batch_size_per_step", "second_level_tokenizer"]
            )
        self.args = training_args
        self.adhoc_args = adhoc_args

        embedding_size = self._get_second_level_tokenizer_dim(second_level_tokenizer)
        self.d_model = embedding_size
        self.d_hid = embedding_size
        self.dropout = dropout
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_sequence_len = self.args.max_sequence_length
        self.mask_scheme = self.args.mask_scheme
        if not self.args.use_custom_roberta_config:
            self.bert = AutoModel.from_pretrained(f"sentence-transformers/{self.args.encoder_name}")
            self.bert_configuration = AutoConfig.from_pretrained(
                f"sentence-transformers/{self.args.encoder_name}"
            )
            self.bert.pooler = None
            self.bert.embeddings.word_embeddings = None

        else:
            self.bert_configuration = AutoConfig.from_pretrained(
                f"sentence-transformers/{self.args.encoder_name}"
            )
            self.bert = AutoModel.from_config(self.bert_configuration)
            self.args.max_sequence_length = self.max_sequence_len
            self.bert.pooler = None
            self.bert.embeddings.word_embeddings = None
        self.cls = SimpleClsHead(self.bert_configuration)
        self.bert.embeddings.position_embeddings = torch.nn.Embedding(self.max_sequence_len, self.bert_configuration.hidden_size)

        # 0 is padding, 1 is [CLS], 2 is [SEP], 3 is [MASK]
        self.special_vec_embeddings = nn.Embedding(
            num_embeddings=4, embedding_dim=embedding_size, padding_idx=0
        )

        if self.args.loss_func == "cosine":
            self.mlm_loss_func = nn.CosineEmbeddingLoss(margin=0.2, reduction="none")
        elif self.args.loss_func == "l1":
            self.mlm_loss_func = nn.L1Loss(reduction="none")
        elif self.args.loss_func == "l2":
            self.mlm_loss_func = nn.MSELoss(reduction="none")
        elif self.args.loss_func == "smoothl1":
            self.mlm_loss_func = nn.SmoothL1Loss(beta=1, reduction="none")
        else:
            self.mlm_loss_func = nn.CosineEmbeddingLoss(reduction="none")

        self.metric = torch.nn.CosineSimilarity(dim=-1)
        self.save_hyperparameters(ignore=["second_level_tokenizer"])
        self.val_step_outputs = []
        self.test_step_outputs = []

    def _get_second_level_tokenizer_dim(self, second_level_tokenizer: SentenceTransformer):
        return next(second_level_tokenizer.parameters()).shape[1]

    def forward(self, batch):
        masked_input_embeddings = self.insert_special_vectors(
            batch["input_embeddings"], batch["masks"], batch["special_vec_lookup"]
        )

        inputs = {
            "token_type_ids": batch["segment_ids"],
            "attention_mask": batch["attention_mask"],
            "inputs_embeds": masked_input_embeddings,
            "output_hidden_states": True,
            "return_dict": False,
        }
        outputs = self.bert(**inputs)
        sequence_output, _, hidden_states = outputs[:3]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores, sequence_output, hidden_states[1]


    # this custom loss function is necessary to calculate the loss per sample, since the number of masks can be different for each sample
    def within_sample_loss(self, prediction_scores, targets, masking):
        if self.args.loss_func == "cosine":
            tmp = prediction_scores.view(-1, prediction_scores.shape[-1])
            tmp2 = targets.view(-1, targets.shape[-1])
            loss_per_mask = self.mlm_loss_func(
                tmp, tmp2, target=torch.ones(len(tmp), device=tmp.device)
            )
            loss_per_mask = loss_per_mask.view(
                prediction_scores.shape[0], prediction_scores.shape[1]
            )
        else:
            loss_per_mask = self.mlm_loss_func(prediction_scores, targets).mean(dim=-1)
        denom = torch.sum(masking, -1, keepdim=True)
        loss_per_sample = torch.sum(loss_per_mask * masking, dim=1).unsqueeze(-1) / denom
        return loss_per_sample.mean()

    def insert_special_vectors(self, sentences, masks, special_vec_lookup):
        """
        Insert special (pad, cls, sep, mask) vectors into the batch. Each special vec type has its own mask to reidentify the positions to insert.
        Can't do this during collating since collating uses multiprocessing and that can't deal with the embeddings lying on the GPU. You get a cuda spawn error.
        """
        masking_index = masks["masked_indices"]
        sep_mask = masks["sep_mask"]
        cls_mask = masks["cls_mask"]
        padding_mask = masks["padding_mask"]
        special_vectors = self.special_vec_embeddings(special_vec_lookup)
        all_masks = torch.stack((list(masks.values()))[1:]).sum(dim=0)
        # assert that all values in all_masks are either 0 or 1 (that the different masks do not overlap)
        assert torch.all(all_masks <= 1)
        masked_sentences = sentences * (all_masks.unsqueeze(-1) != 1)
        if self.mask_scheme == "original_bert":
            masked = masks["masked"]
            replace = masks["replace"]
            masked_sentences = (
                masked_sentences
                + special_vectors[3].unsqueeze(0) * masked.unsqueeze(-1)
                + special_vectors[2].unsqueeze(0) * sep_mask.unsqueeze(-1)
                + special_vectors[1].unsqueeze(0) * cls_mask.unsqueeze(-1)
                + special_vectors[0].unsqueeze(0) * padding_mask.unsqueeze(-1)
            )
            tmp1 = sentences.view(-1, sentences.shape[-1])
            idx = torch.randperm(tmp1.shape[0])
            rand_sents = tmp1[idx].view(sentences.shape)
            masked_sentences = masked_sentences + replace.unsqueeze(-1) * rand_sents
        else:
            masked_sentences = (
                masked_sentences
                + special_vectors[3].unsqueeze(0) * masking_index.unsqueeze(-1)
                + special_vectors[2].unsqueeze(0) * sep_mask.unsqueeze(-1)
                + special_vectors[1].unsqueeze(0) * cls_mask.unsqueeze(-1)
                + special_vectors[0].unsqueeze(0) * padding_mask.unsqueeze(-1)
            )
        return masked_sentences

    def training_step(self, batch):
        input_embeddings = batch["input_embeddings"]
        masks = batch["masks"]
        out = self(batch)
        prediction_scores, sequence_output, hidden_states = out
        mlm_loss = self.within_sample_loss(
            prediction_scores, input_embeddings, masks["masked_indices"]
        )
        hidden_states = hidden_states.detach()
        sequence_output = sequence_output.detach()
        prediction_scores = prediction_scores.detach()
        loss = mlm_loss

        self.log_dict(
            {
                "train/loss": loss.detach(),
                "train/predictions_targets_sim": self.metric(
                    prediction_scores[:, 4], input_embeddings[:, 4]
                ).mean(),  # uses this representative position in the sequence to log cosine similarity of predictions and targets as a sanity check
            },
            on_step=True,
            add_dataloader_idx=False,
            batch_size=self.args.batch_size_per_device,
        )
        return loss

    def validation_step(self, val_batch):
        input_embeddings = val_batch["input_embeddings"]
        masks = val_batch["masks"]
        out = self(val_batch)
        prediction_scores, sequence_output, hidden_states = out
        mlm_loss = self.within_sample_loss(
            prediction_scores, input_embeddings, masks["masked_indices"]
        )
        hidden_states = hidden_states.detach()
        sequence_output = sequence_output.detach()
        prediction_scores = prediction_scores.detach()
        loss = mlm_loss
        self.log_dict(
            {
                "val/loss": loss.detach(),
                "val/predictions_targets_sim": self.metric(
                    prediction_scores[:, 4], input_embeddings[:, 4]
                ).mean(),
            },
            on_step=False,
            add_dataloader_idx=False,
            sync_dist=True,
            batch_size=self.args.batch_size_per_device,
        )
        return loss

    def test_step(self, test_batch):
        input_embeddings = test_batch["input_embeddings"]
        masks = test_batch["masks"]
        out = self(test_batch)
        prediction_scores, sequence_output, hidden_states = out
        mlm_loss = self.within_sample_loss(
            prediction_scores, input_embeddings, masks["masked_indices"]
        )
        hidden_states = hidden_states.detach()
        sequence_output = sequence_output.detach()
        prediction_scores = prediction_scores.detach()
        loss = mlm_loss

        self.log_dict(
            {
                "test/loss": loss.detach(),
                "test/predictions_targets_sim": self.metric(
                    prediction_scores[:, 4], input_embeddings[:, 4]
                ).mean(),
            },
            on_step=False,
            add_dataloader_idx=False,
            batch_size=self.args.batch_size_per_device,
        )
        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay}"
            )

        named_parameters = (
            list(self.bert.named_parameters())
            + list(self.cls.named_parameters())
            + list(self.special_vec_embeddings.named_parameters())
        )

        ### Filter out parameters that are not optimized (requires_grad == False)

        named_parameters = list(
            filter(lambda named_param: named_param[1].requires_grad, named_parameters)
        )

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.args.learning_rate,
            betas=(self.adhoc_args.adam_beta1, self.adhoc_args.adam_beta2),
            eps=self.adhoc_args.adam_epsilon,
        )
        chunking_factor = 16 / self.args.chunking if self.args.chunking > 0 else 24 # this calculation assumes that sentence-based chunking results in chunks of 24 tokens on average
        total_steps = int(self.args.max_epochs * 21000 * chunking_factor)  # hard-coded number of batches per epoch for sentence based chunking
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.lr_warmup * total_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def only_mask_vector(
        self,
        masking_rate: float,
        sentences: torch.Tensor,
        sep_mask: torch.Tensor,
        cls_mask: torch.Tensor,
    ):
        bernou = torch.distributions.bernoulli.Bernoulli(masking_rate)
        bernou_sample = bernou.sample(sample_shape=(sentences.shape[0], sentences.shape[1]))
        # never mask special tokens
        masked_indices = bernou_sample * (sep_mask == 0) * (cls_mask == 0)
        return masked_indices

    def mask_like_bert(
        self,
        masking_rate: float,
        sentences: torch.Tensor,
        sep_mask: torch.Tensor,
        cls_mask: torch.Tensor,
    ):
        bernou = torch.distributions.bernoulli.Bernoulli(masking_rate)
        bernou_sample = bernou.sample(sample_shape=(sentences.shape[0], sentences.shape[1]))
        # never mask special tokens
        selected = bernou_sample * (sep_mask == 0) * (cls_mask == 0)
        # mask 80% of the selected tokens
        masked = selected * (torch.rand_like(selected) < 0.8)
        # replace 10% of the selected tokens with random tokens
        replace = selected * (masked == 0) * (torch.rand_like(selected) < 0.5)
        # keep 10% of the selected tokens
        # keep = selected * (masked == 0) * (replace == 0)
        return selected, masked, replace

    def pretraining_collate(self, examples, masking_rate: float = 0.15):
        masks = {}
        sentences = examples["embeddings"].view(
            self.args.batch_size_per_device, self.args.max_sequence_length - 1, -1
        )
        sep_mask = torch.where(sentences == 0, 1, 0)[
            :, :, 0
        ]  # mask for the separator token, one where the separator token is, 0 everywhere else
        sentences = torch.cat(
            [torch.zeros((sentences.shape[0], 1, sentences.shape[2]), dtype=torch.long), sentences],
            dim=1,
        )
        sep_mask = torch.cat(
            [torch.zeros((sentences.shape[0], 1), dtype=torch.long), sep_mask], dim=1
        )
        padding_mask = torch.zeros_like(sep_mask, dtype=torch.long)
        cls_mask = torch.zeros_like(sep_mask, dtype=torch.long)
        cls_mask[:, 0] = 1
        if self.mask_scheme == "original_bert":
            masked_indices, masked, replace = self.mask_like_bert(
                masking_rate, sentences, sep_mask, cls_mask
            )
            masks["masked_indices"] = masked_indices
            masks["masked"] = masked
            masks["replace"] = replace
        else:
            masked_indices = self.only_mask_vector(masking_rate, sentences, sep_mask, cls_mask)
            masks["masked_indices"] = masked_indices
        special_vec_lookup = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        segment_ids = torch.zeros((sentences.shape[0], sentences.shape[1]), dtype=torch.long)
        attention_mask = torch.ones((sentences.shape[0], sentences.shape[1])).long()
        masks["sep_mask"] = sep_mask
        masks["cls_mask"] = cls_mask
        masks["padding_mask"] = padding_mask
        batch = {
            "input_embeddings": sentences,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "masks": masks,
            "special_vec_lookup": special_vec_lookup,
        }
        return batch


class CustomizedSBERT(SentenceTransformer):
    """
    An adapted version of the SentenceTransformer class from the sentence-transformers package. Enables multi-GPU encoding on already tokenized chunks.
    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME enviroment variable.
    :param use_auth_token: HuggingFace authentication token to download private models.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[nn.Module]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        use_auth_token: Union[bool, str, None] = None,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )

    def encode(
        self,
        input_ids: List[torch.Tensor],
        batch_size: int = 32,
        show_progress_bar=False,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        # if isinstance(input_ids, str) or not hasattr(input_ids, '__len__'):  #Cast an individual sentence to a list with length 1
        #    input_ids = [input_ids]
        #    input_was_string = True

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []
        # length_sorted_idx = np.argsort([-len(sen) for sen in input_ids])
        # sentences_sorted = [input_ids[idx] for idx in length_sorted_idx]
        sentences_sorted = input_ids

        for start_index in trange(
            0, len(input_ids), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = sentence_transformers.util.batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    # if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        # all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings).cpu()
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def tokenize(self, sentences_batch):
        attention_mask = torch.ones_like(sentences_batch)
        attention_mask[sentences_batch == self.tokenizer.pad_token_id] = 0
        return {"input_ids": sentences_batch, "attention_mask": attention_mask}

    def encode_multi_process(
        self,
        input_ids: List[torch.Tensor],
        pool: Dict[str, object],
        batch_size: int = 32,
        chunk_size: int = None,
    ):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(input_ids) / len(pool["processes"]) / 10), 5000)

        logger.debug(
            f"Chunk data into {math.ceil(len(input_ids) / chunk_size)} packages of size {chunk_size}"
        )

        input_queue = pool["input"]
        last_chunk_id = 0

        for i in range(0, len(input_ids), chunk_size):
            chunk = input_ids[i : i + chunk_size]
            input_queue.put([last_chunk_id, batch_size, chunk])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0]
        )
        embeddings = torch.concatenate([result[1] for result in results_list], dim=0)
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                id, batch_size, input_ids = input_queue.get()
                embeddings = model.encode(
                    input_ids,
                    device=target_device,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    batch_size=batch_size,
                )
                results_queue.put([id, embeddings])
            except queue.Empty:
                break

    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ["cpu"] * 4

        logger.info(
            "Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices)))
        )

        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=CustomizedSBERT._encode_multi_process_worker,
                args=(cuda_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}


def _get_normalized_accuracy(num_classes, softmax_dim=-1):
    def _apply_softmax(x):
        return torch.softmax(x, dim=softmax_dim)

    if num_classes is not None and num_classes > 2:
        acc_metric = Accuracy(task="multiclass", num_classes=num_classes)
        input_modifier = _apply_softmax
    else:
        acc_metric = BinaryAccuracy()
        input_modifier = torch.sigmoid

    def _normalized_accuracy(predictions, targets):
        return acc_metric(input_modifier(predictions), targets)
    # acc_metric needs to be assigned to an attribute in self to ensure it is moved
    # to the correct device whenever self is moved to a different device
    return _normalized_accuracy, acc_metric


# based on https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
def sequence_pooling(model_output, attention_mask, aggregator='mean'):
        token_embeddings = (
            model_output  # First element of model_output contains all token embeddings
        )
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        if aggregator == 'mean':
            return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
                )
        elif aggregator == 'max':
            embeddings_masked = torch.where(input_mask_expanded != 0, token_embeddings, float("-inf"))
            return torch.max(embeddings_masked, dim=1)[0]
        else:
            raise ValueError(f"No aggregator '{aggregator}' implemented for pooling! Use one of ['mean', 'max'].")


class DownstreamModel(L.LightningModule):
    def __init__(self, args, misc_args, second_level_tokenizer, model=None):
        super().__init__()
        self.adhoc_args = ModelArgs()
        self.args = args
        self.misc_args = misc_args
        self.save_hyperparameters(
                ignore=["effective_batch_size_per_step", "second_level_tokenizer"]
            )
        if self.args.evaluate_downstream and model is not None:
            model.mask_scheme = "only_mask"
        self.second_level_tokenizer = second_level_tokenizer
        self.model = model
        if model is not None:
            for param in model.cls.parameters():
                param.requires_grad = False
        self.cls_hidden_size = 768  # to match size of longformer cls head
        self.model_embedding_size = self._get_second_level_tokenizer_dim(second_level_tokenizer)
        if self.args.eval_dataset == "quality":
            self.output_size = 4
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif self.args.eval_dataset == "ghomasHudson___muld":
            self.output_size = 1
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.output_size = None
            self.loss_fn = None
        self.max_sequence_len = model.max_sequence_len if model is not None else 512
        self.cls = self._construct_cls()
        self.acc_metric, self._acc_metric_module = _get_normalized_accuracy(self.output_size)
        self.test_predictions = []

    def _get_second_level_tokenizer_dim(self, second_level_tokenizer: CustomizedSBERT):
        return next(second_level_tokenizer.parameters()).shape[1]

    def _construct_cls(self):
        if self.output_size is not None:
            # for quality the 4 answer options are passed through the cls as single examples
            # e.g [4*batch_size, :, 1], not [batch_size, :, 4], hence do not use self.output_size
            # they are reshaped later before applying loss and accuracy computation
            if self.args.eval_dataset == "quality":
                output_size = 1
            else:
                output_size = self.output_size

            if self.args.aggregate in ["all", "concat"]:
                embedding_size = 2 * self.model_embedding_size
            else:
                embedding_size = self.model_embedding_size
            return torch.nn.Sequential(
                torch.nn.Linear(embedding_size, self.cls_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.cls_hidden_size, output_size),
            )
        else:
            return None


    def forward(self, batch):
        if self.model is None:
            raise ValueError('No model was provided during initialization, cannot call forward!')
        out = self.model(batch)
        out = out[1]
 
        if self.output_size is not None:
            if self.args.aggregate in ["all", "concat"]:
                # mean pooling over all tokens except the cls token and the sep token
                document_vector = sequence_pooling(
                    out[:, 1:],
                    (batch["attention_mask"]*(batch["masks"]['sep_mask']==0))[:, 1:],
                    aggregator=self.args.pooling_aggregator,
                    )
                sbert_extra_chunk = self.second_level_tokenizer.encode(batch["input_ids_qa"],
                                                                 convert_to_tensor=True,
                                                                 device=self.device,
                                                                 ).to(self.device)
            if self.args.aggregate in ["last", "concat"]:
                # create new mask where last token in sequence before sep and padding tokens is masked
                tmp_mask = torch.zeros_like(batch["attention_mask"],
                                            device=batch["input_embeddings"].device,
                                            dtype=torch.bool,
                                            )
                tmp_mask[batch["masks"]["sep_mask"] == 1] = 1
                tmp_mask = tmp_mask[:, 1:]
                tmp_mask = torch.cat(
                    [tmp_mask,
                     torch.zeros((tmp_mask.shape[0], 1),
                                 dtype=torch.long,
                                 device=batch["attention_mask"].device,
                                 )
                     ],
                    dim=1,
                )
                # choose only the last token (should be the extra chunk) in the sequence before sep and padding tokens
                out1 = tmp_mask.unsqueeze(-1) * out
                extra_chunk = torch.sum(out1, dim=1)
            
            match self.args.aggregate:
                case "all":
                    out = torch.cat([document_vector, sbert_extra_chunk], dim=1)
                case "last":
                    out = extra_chunk
                case "concat":
                    out = torch.cat([document_vector, extra_chunk], dim=1)
                case "cls":
                    out = batch["input_embeddings"][:, 0]
                case _:
                    raise ValueError("Argument aggregation must be one of ['all', 'last', 'concat', cls] "
                                     f"but is {self.args.aggregation}!")

            out = self.cls(out)
        else:
            # mean or max pooling over all tokens except the cls token and the sep token
            out = sequence_pooling(out[:, 1:],
                                   (batch["attention_mask"]*(batch["masks"]['sep_mask']==0))[:, 1:],
                                   aggregator=self.args.pooling_aggregator,
                                   )
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        if self.loss_fn is not None:
            if self.args.eval_dataset == "quality":
                out = out.view(-1, 4)
                labels = batch["labels"].view(out.shape)[:, 0]
                batch_size = self.args.batch_size_per_device // 4
            else:
                labels = batch["labels"].view_as(out).float()
                batch_size = self.args.batch_size_per_device
            loss = self.loss_fn(out, labels)
            acc = self.acc_metric(out, labels)
            self.log_dict(
                {
                    "train/loss": loss.detach(),
                    "train/accuracy": acc.detach(),
                },
                on_step=True,
                add_dataloader_idx=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            return loss
        else:
            return None

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if self.loss_fn is not None:
            if self.args.eval_dataset == "quality":
                out = out.view(-1, 4)
                labels = batch["labels"].view(out.shape)[:, 0]
                batch_size = self.args.batch_size_per_device // 4
            else:
                labels = batch["labels"].view_as(out).float()
                batch_size = self.args.batch_size_per_device
            loss = self.loss_fn(out, labels)
            acc = self.acc_metric(out, labels)
            self.log_dict(
                {
                    "val/loss": loss.detach(),
                    "val/accuracy": acc.detach(),
                },
                on_step=False,
                add_dataloader_idx=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            return loss
        else:
            return None

    def test_step(self, batch, batch_idx):
        out = self(batch)
        if self.loss_fn is not None:
            if self.args.eval_dataset == "quality":
                out = out.view(-1, 4)
                labels = batch["labels"].view(out.shape)[:, 0]
                batch_size = self.args.batch_size_per_device // 4
            else:
                labels = batch["labels"].view_as(out).float()
                batch_size = self.args.batch_size_per_device
            loss = self.loss_fn(out, labels)
            acc = self.acc_metric(out, labels)
            self.log_dict(
                {
                    "test/loss": loss.detach(),
                    "test/accuracy": acc.detach(),
                },
                on_step=False,
                add_dataloader_idx=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.test_predictions.append(out.detach().cpu())
            return loss
        else:
            self.test_predictions.append(out.detach().cpu())
            return None

    def on_test_epoch_end(self):
        predictions = torch.cat(self.test_predictions, dim=0)
        adapted_checkpoint_path = re.search(r"model(.*)\/", self.args.checkpoint_path).group(0)
        path = os.path.join(
            self.misc_args.output_dir, f"{self.args.eval_dataset}", adapted_checkpoint_path
        )
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(predictions, os.path.join(path, "predictions.pt"))
        results = self.trainer.datamodule.dataset.evaluate(
            predictions, self.trainer.datamodule.downstream_dataset["test"]
        )
        print(results)
        self.log_dict({"test/" + metric + "/full_epoch": value for metric, value in results.items()})

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay}"
            )
        if self.model is None:
            named_parameters = list(self.cls.named_parameters())
        elif self.args.only_tune_cls_head:
            named_parameters = list(self.cls.named_parameters())
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            named_parameters = list(self.model.named_parameters()) + list(
                self.cls.named_parameters()
            )

        ### Filter out parameters that are not optimized (requires_grad == False)

        named_parameters = list(
            filter(lambda named_param: named_param[1].requires_grad, named_parameters)
        )

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.args.learning_rate,
            betas=(self.adhoc_args.adam_beta1, self.adhoc_args.adam_beta2),
            eps=self.adhoc_args.adam_epsilon,
        )
        if self.trainer.datamodule.train_dataloader() is None:
            return optimizer
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.lr_warmup * self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def downstream_collate(self, examples):
        masks = {}
        examples = {key: [example[key] for example in examples] for key in examples[0]}
        doc_data = self.trainer.datamodule.downstream_dataset[
                _extract_trainer_stage(self.trainer.datamodule.trainer.state.stage)
                if (self.trainer.datamodule.trainer.state.stage not in ["sanity_check", "validate"])
                else "validation"
            ].select(range(examples["idx"][0], examples["idx"][-1]+1))
        sentences, sep_mask, padding_mask = self.get_sentence_sequences(
            examples,
            self.trainer.datamodule.sentences_per_split[
                _extract_trainer_stage(self.trainer.datamodule.trainer.state.stage)
                if (self.trainer.datamodule.trainer.state.stage not in ["sanity_check", "validate"])
                else "validation"
            ],
        )
        sentences = torch.cat(
            [torch.zeros((sentences.shape[0], 1, sentences.shape[2]), dtype=torch.long), sentences],
            dim=1,
        )
        sep_mask = torch.cat(
            [torch.zeros((sentences.shape[0], 1), dtype=torch.long), sep_mask], dim=1
        )
        padding_mask = torch.cat(
            [torch.zeros((sentences.shape[0], 1), dtype=torch.long), padding_mask], dim=1
        )
        cls_mask = torch.zeros_like(sep_mask)
        cls_mask[:, 0] = 1
        masked_indices = torch.zeros_like(sep_mask)
        special_vec_lookup = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        segment_ids = torch.zeros((sentences.shape[0], sentences.shape[1]), dtype=torch.long)
        attention_mask = padding_mask == 0
        masks["masked_indices"] = masked_indices
        masks["sep_mask"] = sep_mask
        masks["cls_mask"] = cls_mask
        masks["padding_mask"] = padding_mask
        batch = {
                "input_embeddings": sentences, 
                "segment_ids": segment_ids, 
                "attention_mask": attention_mask, 
                "masks": masks, 
                "special_vec_lookup": special_vec_lookup, 
                "titles": doc_data["document_title"]
        }
        if "label" in doc_data.column_names:
            if self.args.eval_dataset == "ghomasHudson___muld":
                assert all([isinstance(example, int) and example >= 0 for example in examples["label"]]), \
                    "Invalid label encountered, should be positive int!"
                batch["labels"] = torch.tensor(examples["label"])
                batch["input_ids_qa"] = self.second_level_tokenizer.tokenizer(
                    doc_data["qa"], padding=True, truncation=True, add_special_tokens=False, return_tensors="pt"
                    )['input_ids']
            elif self.args.eval_dataset == "quality":
                text_qa = [" ".join([doc_data["question"][i], doc_data["candidate"][i]]) for i in range(len(doc_data))]
                batch["input_ids_qa"] = self.second_level_tokenizer.tokenizer(text_qa, padding=True, truncation=True, return_tensors="pt")['input_ids']
                labels = torch.tensor(doc_data["label"], dtype=torch.long)
                labels = labels - 1
                batch["labels"] = labels
        return batch

    def get_sentence_sequences(self, batch_ids, sent_dataset):
        # from each document in the batch, retrieve the sequence of text chunk vectors
        # the indeces of the first and last chunk vector in the document are saved in batch_ids
        # sent_dataset holds all text chunk vectors of a dataset (memory-mapped)
        adjusted_seq_len = self.max_sequence_len - 1  # leave one input space per sample for cls
        begin_id = torch.tensor(batch_ids["start_id"])
        end_id = torch.tensor(batch_ids["end_id"])
        if isinstance(sent_dataset, torch.Tensor):
            sent_dataset = sent_dataset
        else:
            sent_dataset = sent_dataset.select(range(begin_id.min(), end_id.max()+1)).with_format(  # end_id is the last sentence id in the document, so we need to add 1 to range call since it is exclusive
                "torch"
            )["embeddings"]
            offset = begin_id.min()
            begin_id = begin_id - offset
            end_id = end_id - offset
        sent_dataset = torch.cat(
            [sent_dataset, torch.zeros((1, sent_dataset.shape[-1]), dtype=torch.long), -1 * torch.ones((1, sent_dataset.shape[-1]), dtype=torch.long)], dim=0
        )
        # prepare index tensor for torch gather to get the sentences for each sample in the batch from the sentence dataset
        # we want the tensor to be of shape (batch_size*adjusted_seq_len, embedding_dim)
        # so that afterwards we can reshape the batch dimension back to (batch_size, adjusted_seq_len, embedding_dim)
        # in the index tensor all columns need to have the same value
        # then blocks of adjusted_seq_len constitute one sample in the batch

        # start with one sample in the batch
        indices = []
        for i in range(len(begin_id)):
            tmp = torch.ones(
                adjusted_seq_len, sent_dataset.shape[-1], dtype=torch.long
            ) * torch.arange(
                begin_id[i], begin_id[i] + adjusted_seq_len, dtype=torch.long
            ).unsqueeze(
                -1
            )
            if tmp[-1, 0] < end_id[i]:
                tmp[-1] = end_id[i]
            tmp = torch.where(tmp > end_id[i], len(sent_dataset) - 1, tmp)
            indices.append(tmp)
        x = torch.vstack(indices)
        index = x
        assert not torch.any(index >= len(sent_dataset))
        sampled_sentences = torch.gather(sent_dataset, 0, index.long())
        # reshape the batch dimension back to (batch_size, adjusted_seq_len, embedding_dim)
        # since at inference the documents have different lengths, we need to reconstruct the sequence length dimension
        sampled_sentences = sampled_sentences.view(len(begin_id), adjusted_seq_len, -1)
        x_sep_mask = torch.where(sampled_sentences == 0, 1, 0)[
            :, :, 0
        ]
        x_padding_mask = torch.where(sampled_sentences == -1, 1, 0)[:, :, 0]
        return sampled_sentences, x_sep_mask, x_padding_mask
