from typing import Mapping
import pytorch_lightning as L
import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig, AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer
from .model import CustomizedSBERT
import nltk
nltk.download('punkt')

class NextLevelBERTConfig(PretrainedConfig):
    def __init__(
        self,
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=256,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        loss_func="smoothl1",
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, cls_token_id=cls_token_id, sep_token_id=sep_token_id, mask_token_id=mask_token_id, **kwargs)
        self._encoder_name = encoder_name
        if not isinstance(chunk_size, int) or chunk_size < 0:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        self.encoder_config = AutoConfig.from_pretrained(self.encoder_name)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.loss_func = loss_func
    
    @property
    def encoder_name(self) -> str:
        return getattr(self, "_encoder_name", None)
    
    @encoder_name.setter
    def encoder_name(self, value):
        self._encoder_name = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

class NextLevelBERT(L.LightningModule, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = NextLevelBERTConfig(**config)
        self.config = config
        self.encoder = CustomizedSBERT(config.encoder_name)
        self.bert = AutoModel.from_pretrained(self.config.encoder_name)
        self.bert.pooler = None
        self.bert.embeddings.word_embeddings = None
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.model_embedding_size = self._get_second_level_tokenizer_dim(self.encoder)
        self.max_sequence_len = self.config.encoder_config.max_position_embeddings
        if self.config.encoder_config.model_type == "roberta":
            self.max_sequence_len -= 2
        
        self.special_vec_embeddings = nn.Embedding(
            num_embeddings=4, embedding_dim=self.model_embedding_size, padding_idx=0
        )

        if self.config.loss_func == "cosine":
            self.mlm_loss_func = nn.CosineEmbeddingLoss(margin=0.2, reduction="none")
        elif self.config.loss_func == "l1":
            self.mlm_loss_func = nn.L1Loss(reduction="none")
        elif self.config.loss_func == "l2":
            self.mlm_loss_func = nn.MSELoss(reduction="none")
        elif self.config.loss_func == "smoothl1":
            self.mlm_loss_func = nn.SmoothL1Loss(beta=1, reduction="none")
        else:
            self.mlm_loss_func = nn.CosineEmbeddingLoss(reduction="none")

    def _get_second_level_tokenizer_dim(self, second_level_tokenizer: SentenceTransformer):
        return next(second_level_tokenizer.parameters()).shape[1]

    def load_state_dict_from_artifact(self, artifact):
        self.load_state_dict(artifact["state_dict"], strict=False)

    def forward(self, batch):
        """
        Expects a batch dictionary with the following keys:
        - input_embeddings: torch.Tensor of shape (batch_size, sequence_length, model_embedding_size)
        - masks: a dictionary with masks (1s where the special vector is, 0 otherwise) for the following
            - masked_indices: torch.Tensor of shape (batch_size, sequence_length): 1 where the vector should be replaced by the mask vector
            - sep_mask: torch.Tensor of shape (batch_size, sequence_length): 1 where the sep token is. should be at the end of each document
            - cls_mask: torch.Tensor of shape (batch_size, sequence_length): 1 where the cls token is. should be at the beginning of each document
            - padding_mask: torch.Tensor of shape (batch_size, sequence_length): 1 where the padding vector is
        """
        masked_input_embeddings = self.insert_special_vectors(
            batch["input_embeddings"], batch["masks"]
        )

        inputs = {
            "token_type_ids": batch["segment_ids"],  # TODO check how this is used in roberta
            "attention_mask": batch["masks"]["attention_mask"],  # TODO should be one everywhere if no padding present
            "inputs_embeds": masked_input_embeddings,
            "output_hidden_states": True,
            "return_dict": False,
        }
        outputs = self.bert(**inputs)
        sequence_output, _, hidden_states = outputs[:3]
        return sequence_output


    def insert_special_vectors(self, sentences, masks):
        """
        Insert special (pad, cls, sep, mask) vectors into the batch. Each special vec type has its own mask to reidentify the positions to insert.
        Can't do this during collating since collating uses cpu multiprocessing and that can't deal with the embeddings lying on the GPU. You get a cuda spawn error.
        """
        special_vec_lookup = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        sep_mask = masks["sep_mask"]
        cls_mask = masks["cls_mask"]
        padding_mask = masks["padding_mask"]
        special_vectors = self.special_vec_embeddings(special_vec_lookup)
        mask_keys = list(masks.keys())
        if "attention_mask" in mask_keys:
            mask_keys.remove("attention_mask") # remove this mask at this point because it has some redundant information to the other masks and will result in all_masks being > 1
        all_masks = torch.stack([masks[key] for key in mask_keys]).sum(dim=0)
        # assert that all values in all_masks are either 0 or 1 (that the different masks do not overlap)
        assert torch.all(all_masks <= 1)
        masked_sentences = sentences * (all_masks.unsqueeze(-1) != 1)

        masked_sentences = (
            masked_sentences
            + special_vectors[2].unsqueeze(0) * sep_mask.unsqueeze(-1)
            + special_vectors[1].unsqueeze(0) * cls_mask.unsqueeze(-1)
            + special_vectors[0].unsqueeze(0) * padding_mask.unsqueeze(-1)
        )
        return masked_sentences


    def sequence_pooling(self, model_output, attention_mask):
        chunk_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(chunk_embeddings.size()).float()
        return torch.sum(chunk_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
            )


    def _chunk_examples(self, examples):
        chunk_size = self.config.chunk_size
        chunks = []
        doc_lens = []
        for text in examples:
            chunked_text = self._chunk_example(text, chunk_size=chunk_size)
            # separate documents with a placeholder for sep vector
            chunked_text.append([0] * chunk_size)
            doc_lens.append(len(chunked_text))
            chunks.extend(chunked_text)
        return torch.tensor(chunks), doc_lens


    def _chunk_example(self, text, chunk_size):
        chunks = []
        actual_chunk_size = chunk_size - 2
        for i in range(0, len(text), actual_chunk_size):
            chunk = text[i : i + actual_chunk_size]
            chunk.insert(0, self.encoder.tokenizer.cls_token_id)
            chunk.append(self.encoder.tokenizer.sep_token_id)
            if len(chunk) < chunk_size:
                chunk.extend([self.encoder.tokenizer.pad_token_id] * (chunk_size - len(chunk)))
            chunks.append(chunk)
        return chunks

    def _create_sentence_chunks(self, texts):
        doc_lens = []
        chunks = []
        for doc in texts:
            doc_chunks = []
            for sent in doc:
                # if an individual sentence is longer than the max sequence length of the model (without cls and sep tokens)
                if len(sent) > self.encoder.max_seq_length - 2:
                    sent = sent[: self.encoder.max_seq_length - 2]
                sent.insert(0, self.encoder.tokenizer.cls_token_id)
                sent.append(self.encoder.tokenizer.sep_token_id)
                if len(sent) < self.encoder.max_seq_length:
                    sent.extend([self.encoder.tokenizer.pad_token_id] * (self.encoder.max_seq_length - len(sent)))
                doc_chunks.append(sent)
            doc_chunks.append([0] * self.encoder.max_seq_length)
            doc_lens.append(len(doc_chunks))
            chunks.extend(doc_chunks)
        return torch.tensor(chunks), doc_lens

        
    def _encode_fixed_chunk(self, examples):
        tokenized_examples = self.encoder.tokenizer(examples, padding=False, add_special_tokens=False)["input_ids"]
        chunked_examples, doc_lens = self._chunk_examples(tokenized_examples)
        return chunked_examples, doc_lens
        
    
    def _encode_by_sentence_boundary(self, examples):
        split_into_sentences = [nltk.tokenize.sent_tokenize(example) for example in examples]
        tokenized_sentences = [self.encoder.tokenizer(doc, padding=False, add_special_tokens=False)["input_ids"] for doc in split_into_sentences]
        chunked_examples, doc_lens = self._create_sentence_chunks(tokenized_sentences)
        return chunked_examples, doc_lens


    def encode(self, examples, output_value="document_embeddings", masks=None, encoder_batch_size=2048):
        if self.config.chunk_size == 0:
            chunked_examples, doc_lens = self._encode_by_sentence_boundary(examples)
        elif self.config.chunk_size > 0:
            chunked_examples, doc_lens = self._encode_fixed_chunk(examples)
        else:
            raise ValueError("chunk_size must be a positive integer for a fixed chunk size or 0 for sentence boundary chunking.")
        encoded_chunks = torch.from_numpy(self.encoder.encode(chunked_examples, batch_size=encoder_batch_size, show_progress_bar=False))
        # reshape as by document boundaries
        encoded_chunks = encoded_chunks.split(doc_lens)
        prepared_inputs = self.prepare_inputs(encoded_chunks, masks=masks)
        # encode with NextLevelBERT
        out = self.forward(prepared_inputs)
        if output_value == "document_embeddings":
            # don't include cls and sep vectors in the pooling
            out = self.sequence_pooling(out[:, 1:],
                                        (prepared_inputs["masks"]["attention_mask"]*(prepared_inputs["masks"]['sep_mask']==0))[:, 1:])
        return out

    
    def prepare_inputs(self, inputs, masks):
        """
        Expects the inputs as a list or tuple of length batch_size with torch.Tensors of shape (sequence_length-1, model_embedding_size). 
        Why sequence_length-1? Because the input is expected to be the sequence without the cls vector at the beginning. The cls vector position is added later in this method.
        Each item in the list corresponds to the tensors of a document.
        Masks can be given via the masks dictionary. If given, they need to be a dict of tensors of shape (batch_size, sequence_length). If not given, the masks are created here.
        """
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)[:,:self.max_sequence_len-1]
        # add a padding vector at the beginning of each sequence as a placeholder for the cls vector (which is added in forward)
        padded_inputs = torch.cat([torch.zeros(padded_inputs.shape[0], 1, padded_inputs.shape[-1]), padded_inputs], dim=1)
        lens = [len(i) if len(i) < self.max_sequence_len else (self.max_sequence_len-1) for i in inputs]
        if masks is None:
            given_masks = []
            masks = {}
        else:
            given_masks = list(masks.keys())
        if "sep_mask" in given_masks:
            masks["sep_mask"] = torch.nn.utils.rnn.pad_sequence(masks["sep_mask"], batch_first=True)
        else:
            masks["sep_mask"] = torch.zeros(padded_inputs.shape[:2], dtype=torch.long)
            # put 1 at the index where the unpadded sequence ends
            masks["sep_mask"] = masks["sep_mask"].scatter(1, (torch.tensor(lens)).unsqueeze(1), torch.ones((padded_inputs.shape[0],1), dtype=torch.long))
        if "cls_mask" in given_masks:
            masks["cls_mask"] = torch.nn.utils.rnn.pad_sequence(masks["cls_mask"], batch_first=True)
        else:
            masks["cls_mask"] = torch.zeros(padded_inputs.shape[:2], dtype=torch.long)
            masks["cls_mask"][:, 0] = 1
        if "padding_mask" in given_masks:
            assert masks["padding_mask"].shape == padded_inputs.shape[:2]
        else:
            masks["padding_mask"] = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=1)[:,:self.max_sequence_len-1]
            masks["padding_mask"] = torch.cat([torch.zeros(masks["padding_mask"].shape[0], 1, masks["padding_mask"].shape[-1]), masks["padding_mask"]], dim=1)
            masks["padding_mask"] = masks["padding_mask"][:,:,0].eq(1).long()
        if "segment_ids" in given_masks:
            segment_ids = torch.nn.utils.rnn.pad_sequence(masks["segment_ids"], batch_first=True)
        else:
            segment_ids = torch.zeros(padded_inputs.shape[:2], dtype=torch.long)
        if "attention_mask" in given_masks:
            masks["attention_mask"] = torch.nn.utils.rnn.pad_sequence(masks["attention_mask"], batch_first=True)
        else:
            masks["attention_mask"] = masks["padding_mask"] != 1
        return {"input_embeddings": padded_inputs, "masks": masks, "segment_ids": segment_ids}
