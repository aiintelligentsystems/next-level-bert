from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from loguru import logger
from torch.optim import AdamW
from torch import nn
import os
from src.model import CustomizedSBERT, _get_normalized_accuracy, sequence_pooling

from transformers import (
    LongformerModel,
    LongformerTokenizer,
    LongformerForMultipleChoice,
    get_cosine_schedule_with_warmup,
)
from src.model import DownstreamModel

if TYPE_CHECKING:
    from train import TrainingArgs


@dataclass
class ModelArgs:
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


class PretrainedModel(L.LightningModule):
    def __init__(self, training_args, misc_args):
        super().__init__()
        if not training_args.resume_training:
            self.save_hyperparameters(ignore=["effective_batch_size_per_step"])
        self.args = training_args
        self.adhoc_args = ModelArgs()
        self.misc_args = misc_args
        self.model = None
        self.model_name = self.args.model_name_or_path
        self.cls_hidden_size = 768
        self.test_step_outputs = []
        if self.args.eval_dataset == "quality":
            self.output_size = 4
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif self.args.eval_dataset == "ghomasHudson___muld":
            self.output_size = 1
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.output_size = None
            self.loss_fn = None
        self.acc_metric, self._acc_metric_module = _get_normalized_accuracy(self.output_size)

    def _construct_cls(self):
        if self.output_size is not None:
            # for quality the 4 answer options are passed through the cls as single examples
            # e.g [4*batch_size, :, 1], not [batch_size, :, 4], hence do not use self.output_size
            # they are reshaped later before applying loss and accuracy computation
            if self.args.eval_dataset == "quality":
                output_size = 1
            else:
                output_size = self.output_size

            # for sentence_transformer models (SBERT) we always use 2 vectors as representation for muld and quality
            if (self.args.model_name_or_path == "PretrainedSBERT"
                    and self.args.eval_dataset in ["quality", "ghomasHudson___muld"]):
                embedding_size = 2*self.model_embedding_size
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
        raise NotImplementedError("Forward pass needs to be implemented in subclass")
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        if self.loss_fn is not None:
            if self.args.eval_dataset == "quality":
                out = out.view(-1, 4)
                labels = batch["label"].view(out.shape)[:, 0]
                batch_size = self.args.batch_size_per_device // 4
            else:
                labels = batch["label"].view_as(out).float()
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
                labels = batch["label"].view(out.shape)[:, 0]
                batch_size = self.args.batch_size_per_device // 4
            else:
                labels = batch["label"].view_as(out).float()
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

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        out = self(batch)
        if self.loss_fn is not None:
            if self.args.eval_dataset == "quality":
                out = out.view(-1, 4)
                labels = batch["label"].view(out.shape)[:, 0]
                batch_size = self.args.batch_size_per_device // 4
            else:
                labels = batch["label"].view_as(out).float()
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
            self.test_step_outputs.append(out.cpu())
            return loss 
        else:
            self.test_step_outputs.append(out.cpu())
            return None
    
    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay}"
            )
        if self.args.only_tune_cls_head:
            named_parameters = list(self.cls.named_parameters())
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            named_parameters = list(self.model.named_parameters()) + list(self.cls.named_parameters())

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

    def on_test_epoch_end(self):
        predictions = torch.cat(self.test_step_outputs, dim=0)
        adapted_checkpoint_path = f"{self.model_name}_{self.args.encoder_name}"
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

    def downstream_collate(self, examples):
        examples = {key: [example[key] for example in examples] for key in examples[0]}
        if self.args.encoder_name == "nomic":
            if self.args.eval_dataset == "booksum_chapter":
                examples["text"]
            examples["text"] = [" ".join(["clustering:", examples["text"][i]]) for i in range(len(examples["text"]))]
        if self.args.eval_dataset == "ghomasHudson___muld":
            assert all([isinstance(example, int) and example >= 0 for example in examples["label"]]), \
                "Invalid label encountered, should be positive int!"
            examples["label"] = torch.tensor(examples["label"])
            examples["input_ids"] = self.tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")["input_ids"]
            examples["input_ids_qa"] = self.tokenizer(examples["qa"], padding=True, truncation=True, return_tensors="pt")['input_ids']
        elif self.args.eval_dataset == "quality":
            examples["text_qa"] = [" ".join([examples["question"][i], examples["candidate"][i]]) for i in range(len(examples["question"]))]
            self.tokenizer.truncation_side = "left"
            examples["input_ids"] = self.tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")['input_ids']
            examples["input_ids_qa"] = self.tokenizer(examples["text_qa"], padding=True, truncation=True, return_tensors="pt")['input_ids']
            labels = torch.tensor(examples["label"], dtype=torch.long)
            labels = labels - 1
            examples["label"] = labels
        else:
            examples["input_ids"] = self.tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")["input_ids"]
        return examples


class PretrainedSBERT(PretrainedModel):
    def __init__(self, training_args, misc_args):
        super().__init__(training_args, misc_args)
        if self.args.encoder_name == "nomic":
            self.model = CustomizedSBERT(
                "nomic-ai/nomic-embed-text-v1", trust_remote_code=True, device="cuda"
            )
        else:
            self.model = CustomizedSBERT(self.args.encoder_name, device=self.device)
        self.tokenizer = self.model.tokenizer
        self.model_embedding_size = self.model._modules["1"].word_embedding_dimension
        self.cls = self._construct_cls()

    def forward(self, batch):
        out = self.model.encode(batch["input_ids"], convert_to_tensor=True, device=self.device)
        if self.args.eval_dataset in ["quality", "ghomasHudson___muld"]:
            qa = self.model.encode(batch["input_ids_qa"], convert_to_tensor=True, device=self.device)
            out = torch.cat([out, qa], dim=1)
        if self.output_size is not None:
            out = self.cls(out.to(self.device))
        return out


class PretrainedLongformer(PretrainedModel):
    def __init__(
        self,
        training_args: "TrainingArgs",  # do in string to remove dependency when loading.
        misc_args,
    ) -> None:
        super().__init__(training_args, misc_args)
        if not training_args.resume_training:
            self.save_hyperparameters(ignore=["effective_batch_size_per_step"])
        self.args = training_args
        self.adhoc_args = ModelArgs()
        self.misc_args = misc_args
        if self.args.eval_dataset == "quality":
            self.model = LongformerForMultipleChoice.from_pretrained("allenai/longformer-base-4096")
            self.output_size = None  # internal head is used, so we set output size to None so that cls is not used
        else:
            self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.model.config.max_global_attn = 64
        # Note: embedding size is looked up for base model and hardcoded
        self.model_embedding_size = 768
        self.cls_hidden_size = 768
        self.cls = self._construct_cls()
        self.test_step_outputs = []

    def forward(self, batch):
        inputs = batch["inputs"]
        if self.args.eval_dataset == "quality":
            inputs["input_ids"] = inputs["input_ids"].view(-1, 4, inputs["input_ids"].shape[-1])
            output_field = "logits"
            if self.cls is not None:
                raise ValueError("Quality dataset uses the internel Longformer cls head, hence self.cls should be None!")
        else:
            global_attention_mask = torch.zeros(
            inputs["input_ids"].shape, dtype=torch.long, device=self.device
            )
            global_attention_mask[:, 0] = 1  # only global attend on CLS token
            inputs["global_attention_mask"] = global_attention_mask
            output_field = "last_hidden_state"
        if self.args.eval_dataset == "ghomasHudson___muld":
            output_field = "cls_embedding"  # just a placeholder value that has no effect
            if self.cls is None:
                raise ValueError("The MULD-movie datset requires self.cls to be initiallized, but it is None!")

        out = self.model(**inputs)

        # note that "all"/"concat" for Longformer are only the pooled output tokens,
        # no second embedding is concatenated
        # use_pooling definition for consistent behavior later in code even if the condition is adjusted
        if use_pooling := (self.args.aggregate in ["all", "concat"]
                           and self.args.eval_dataset != 'quality'):  # skip pooling calculation since quality uses internal cls head
            pooled_embedding = sequence_pooling(out["last_hidden_state"][:, 1:],
                                                batch["inputs"]["attention_mask"][:, 1:],
                                                aggregator=self.args.pooling_aggregator,
                                                )

        # for quality self.output_size == 4, but still self.cls is None because internal CLS head
        # of LongformerForMultipleChoice is used, hence self.cls is used for condition rather than self.output_size
        if self.cls is not None:
            if use_pooling:
                cls_input = pooled_embedding
            else:
                cls_input = out["last_hidden_state"][:, 0]  # first token in sequence is CLS embedding
            return self.cls(cls_input)

        if self.args.eval_dataset == "booksum_chapter":
            if use_pooling:
                return pooled_embedding
            else:
                return out["last_hidden_state"][:, 0]  # first token in sequence is CLS embedding
        return out[output_field]

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay}"
            )
        if self.args.only_tune_cls_head:
            named_parameters = list(self.model.classifier.named_parameters())
            for param in self.model.longformer.parameters():
                param.requires_grad = False
        else:
            named_parameters = list(self.model.named_parameters())

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
        examples = {key: [example[key] for example in examples] for key in examples[0]}
        if self.args.eval_dataset == "ghomasHudson___muld":
            assert all([isinstance(example, int) and example >= 0 for example in examples["label"]]), \
                    "Invalid label encountered, should be positive int!"
            examples["label"] = torch.tensor(examples["label"])
            examples["inputs"] = self.tokenizer(examples["text"], text_pair=examples["qa"], padding=True, truncation="only_first", return_tensors="pt")
        elif self.args.eval_dataset == "quality":
            examples["text_qa"] = [" ".join([examples["question"][i], examples["candidate"][i]]) for i in range(len(examples["question"]))]
            examples["inputs"] = self.tokenizer(text=examples["text"], text_pair=examples["text_qa"], padding=True, truncation="only_first", return_tensors="pt")
            labels = torch.tensor(examples["label"], dtype=torch.long)
            labels = labels - 1
            examples["label"] = labels
        else:
            examples["inputs"] = self.tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")
        return examples


class AvgSBERTSentences(DownstreamModel):
    def __init__(self, training_args, misc_args, second_level_tokenizer):
        super().__init__(training_args, misc_args, second_level_tokenizer)
        if not training_args.resume_training:
            self.save_hyperparameters(ignore=["effective_batch_size_per_step"])
        self.args = training_args
        self.misc_args = misc_args
        self.adhoc_args = ModelArgs()
        embedding_size = self._get_second_level_tokenizer_dim(second_level_tokenizer)
        self.second_level_tokenizer = second_level_tokenizer
        self.max_sequence_len = self.args.max_sequence_length

    def forward(self, batch):
        if self.args.aggregate in ["all", "concat"] or self.args.eval_dataset == "booksum_chapter":
            # mean pooling over all tokens except the cls token and the sep token
            document_vector = sequence_pooling(batch["input_embeddings"][:, 1:],
                                               (batch["attention_mask"]*(batch["masks"]['sep_mask']==0))[:, 1:],
                                               aggregator=self.args.pooling_aggregator,
                                               )
            if self.args.eval_dataset == "booksum_chapter":
                return document_vector
        if self.args.aggregate in ["last", "concat"]:
            extra_chunk = self.second_level_tokenizer.encode(batch["input_ids_qa"], convert_to_tensor=True, device=self.device).to(self.device)

        match self.args.aggregate:
            case "cls":
                out = batch["input_embeddings"][:, 0]
            case "pool":
                out = document_vector
            case "last":
                out = extra_chunk
            case "concat" | "all":
                out = torch.cat([document_vector, extra_chunk], dim=1)
            case _:
                raise ValueError("Argument aggregation must be one of "
                                 f"['all', 'last', 'concat', 'pool'] but is {self.args.aggregation}!"
                                 )
        if self.output_size is not None:
            out = self.cls(out)
        return out

    def on_test_epoch_end(self):
        predictions = torch.cat(self.test_predictions, dim=0)
        adapted_checkpoint_path = (
            "AvgChunks"
        )
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
