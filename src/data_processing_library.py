import re
from datasets import Dataset, DatasetDict
import os
import pandas as pd


def delete_dataset(dataset):
    cached_files = [
        cache_file["filename"]
        for cache_file in dataset.cache_files
        if (
            ("data_splits" not in cache_file["filename"])
            and ("tokenized" not in cache_file["filename"])
        )
    ]
    del dataset
    for cached_file in cached_files:
        os.remove(cached_file)


def clean_str(example):
    text = example["text"]
    text = text.replace("\n", " ")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    for ch in [
        "\\",
        "`",
        "*",
        "_",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        ">",
        "#",
        "+",
        "-",
        "$",
        '"',
        "=",
        "@",
        "%",
        "^",
        "|",
        "~",
        "/",
    ]:
        if ch in text:
            text = text.replace(ch, "")
    text = text.strip()
    text = re.sub(" +", " ", text)
    return {"text": text}


def tokenize_dataset(example, tokenizer, column="text", truncation_side="right"):
    tokenizer_batch_size = 2048
    output_sents = []
    output_masks = []
    input_sents = example[column]
    tokenizer.truncation_side = truncation_side
    for i in range(0, len(input_sents), tokenizer_batch_size):
        output = tokenizer(
            input_sents[i : i + tokenizer_batch_size],
            truncation=False,
            padding=False,
            max_length=None,
            add_special_tokens=False,
        )
        output_sents.extend(output["input_ids"])
        output_masks.extend(output["attention_mask"])
    result = {
        column + "_input_ids": output_sents,
        column + "_attention_mask": output_masks,
        "document_title": example["document_title"],
    }
    if example.get("label") is not None:
        result["label"] = example["label"]
    return result


def create_input_chunks(
    examples,
    idx,
    cls_token_id,
    sep_token_id,
    pad_token_id,
    max_seq_len,
    chunking,
    is_summarization=False,
    is_labeled=False,
    extra_chunk=False,
):
    sentence_counter = 0
    chunks = []
    mask_chunks = []
    sentence_ids = []
    is_summary = []
    summary_id = []
    titles = []
    doc_ids = []
    labels = []
    input_ids = examples["input_ids"] if "input_ids" in examples else examples["text_input_ids"]
    attention_masks = examples["attention_mask"] if "attention_mask" in examples else examples["text_attention_mask"]
    
    for doc_idx, doc_input_ids in enumerate(input_ids):
        doc_input_ids = [item for sublist in doc_input_ids for item in sublist]
        doc_mask = [item for sublist in attention_masks[doc_idx] for item in sublist]
        actual_text_chunk = chunking - 2
        document_counter = idx[doc_idx]
        len_doc = len(doc_input_ids)
        for i in range(0, len_doc, actual_text_chunk):
            chunk = doc_input_ids[i : i + actual_text_chunk]
            mask_chunk = doc_mask[i : i + actual_text_chunk]
            chunk.insert(0, cls_token_id)
            chunk.append(sep_token_id)
            mask_chunk.extend([1, 1])
            if len(chunk) < chunking:
                chunk.extend([pad_token_id] * (chunking - len(chunk)))
                mask_chunk.extend([0] * (chunking - len(mask_chunk)))
            chunks.append(chunk)
            mask_chunks.append(mask_chunk)
            titles.append(examples["document_title"][doc_idx])
            if is_labeled:
                labels.append(examples["label"][doc_idx])
            if is_summarization:
                is_summary.append(examples["is_summary"][doc_idx])
                summary_id.append(examples["summary_id"][doc_idx])
            sentence_ids.append(sentence_counter)
            doc_ids.append(document_counter)
            sentence_counter += 1
        if extra_chunk:  # add extra chunk for question-answer pair in quality
            chunk = examples["qa_input_ids"][doc_idx]
            if len(chunk) > actual_text_chunk:
                chunk = chunk[: actual_text_chunk]
                mask_chunk = mask_chunk[: actual_text_chunk]
            chunk.insert(0, cls_token_id)
            chunk.append(sep_token_id)
            mask_chunk = [1] * len(chunk)
            if len(chunk) < chunking:
                chunk.extend([pad_token_id] * (chunking - len(chunk)))
                mask_chunk.extend([0] * (chunking - len(mask_chunk)))
            chunks.append(chunk)
            mask_chunks.append(mask_chunk)
            sentence_ids.append(sentence_counter)
            titles.append(examples["document_title"][doc_idx])
            if is_labeled:
                labels.append(examples["label"][doc_idx])
            if is_summarization:
                is_summary.append(examples["is_summary"][doc_idx])
                summary_id.append(examples["summary_id"][doc_idx])
            doc_ids.append(document_counter)
            sentence_counter += 1
        chunks.append([0] * (actual_text_chunk + 2))  # insert zero vector after each document. will later be filled by separator vector
        mask_chunks.append([0] * (actual_text_chunk + 2))
        sentence_ids.append(sentence_counter)
        titles.append(examples["document_title"][doc_idx])
        if is_labeled:
            labels.append(examples["label"][doc_idx])
        if is_summarization:
            is_summary.append(examples["is_summary"][doc_idx])
            summary_id.append(examples["summary_id"][doc_idx])
        doc_ids.append(document_counter)
        sentence_counter += 1
    out = {
        "chunks": chunks,
        "masks": mask_chunks,
        "local_sentence_ids": sentence_ids,
        "document_title": titles,
        "doc_ids": doc_ids,
    }
    if is_summarization:
        out["is_summary"] = is_summary
        out["summary_id"] = summary_id
    if is_labeled:
        out["label"] = labels
    return out


def create_input_sents(
    examples,
    idx,
    cls_token_id,
    sep_token_id,
    pad_token_id,
    max_seq_len,
    chunking,
    is_summarization=False,
    is_labeled=False,
    extra_chunk=False,
):
    chunks = []
    titles = []
    labels = []
    doc_ids = []
    mask_chunks = []
    is_summary = []
    summary_id = []
    input_ids = examples["input_ids"] if "input_ids" in examples else examples["text_input_ids"]
    attention_masks = examples["attention_mask"] if "attention_mask" in examples else examples["text_attention_mask"]
    
    for doc_idx, doc_input_ids in enumerate(input_ids):
        for i, item in enumerate(doc_input_ids):
            chunk = item
            mask_chunk = attention_masks[doc_idx][i]
            if len(chunk) > max_seq_len - 2:
                chunk = chunk[: max_seq_len - 2]
                mask_chunk = mask_chunk[: max_seq_len - 2]
            chunk.insert(0, cls_token_id)
            chunk.append(sep_token_id)
            mask_chunk.extend([1, 1])
            # fill up chunk with pad tokens
            if len(chunk) < max_seq_len:
                chunk.extend([pad_token_id] * (max_seq_len - len(chunk)))
                mask_chunk.extend([0] * (max_seq_len - len(mask_chunk)))
            chunks.append(chunk)
            mask_chunks.append(mask_chunk[:max_seq_len])
            titles.append(examples["document_title"][doc_idx])
            if is_labeled:
                labels.append(examples["label"][doc_idx])
            doc_ids.append(idx[doc_idx])
            if is_summarization:
                is_summary.append(examples["is_summary"][doc_idx])
                summary_id.append(examples["summary_id"][doc_idx])
            if extra_chunk:  # add extra chunk for question-answer pair in quality
                chunk = examples["qa_input_ids"][doc_idx]
                if len(chunk) > max_seq_len - 2:
                    chunk = chunk[: max_seq_len - 2]
                chunk.insert(0, cls_token_id)
                chunk.append(sep_token_id)
                mask_chunk = [1] * len(chunk)
                if len(chunk) < max_seq_len:
                    chunk.extend([pad_token_id] * (max_seq_len - len(chunk)))
                    mask_chunk.extend([0] * (max_seq_len - len(mask_chunk)))
                chunks.append(chunk)
                mask_chunks.append(mask_chunk)
                titles.append(examples["document_title"][doc_idx])
                if is_labeled:
                    labels.append(examples["label"][doc_idx])
                if is_summarization:
                    is_summary.append(examples["is_summary"][doc_idx])
                    summary_id.append(examples["summary_id"][doc_idx])
                doc_ids.append(idx[doc_idx])
        chunks.append([0] * max_seq_len)  # insert zero vector after each document
        mask_chunks.append([0] * max_seq_len)
        titles.append(examples["document_title"][doc_idx])
        doc_ids.append(idx[doc_idx])
        if is_labeled:
            labels.append(examples["label"][doc_idx])
        if is_summarization:
            is_summary.append(examples["is_summary"][doc_idx])
            summary_id.append(examples["summary_id"][doc_idx])

    sentence_ids = list(range(len(chunks)))
    out = {
        "chunks": chunks,
        "masks": mask_chunks,
        "local_sentence_ids": sentence_ids,
        "document_title": titles,
        "doc_ids": doc_ids,
    }
    if is_summarization:
        out["is_summary"] = is_summary
        out["summary_id"] = summary_id
    if is_labeled:
        out["label"] = labels
    return out


def create_batch_ids_downstream(dataset, all_test=False):
    splits = ["train", "validation", "test"]
    batch_ids_dict = {}
    if all_test:
        dataset_pd = dataset["all"].to_pandas()
        begin_ids = []
        end_ids = []
        print("Creating batch ids...")
        tmp = (
            dataset_pd.groupby(["document_title", "summary_id"])["sentence_ids"]
            .apply(list)
            .reset_index(name="sentence_ids")
        )
        tmp = tmp.sort_values("sentence_ids", ignore_index=True)
        begin_ids = list(tmp["sentence_ids"].apply(min))
        end_ids = list(tmp["sentence_ids"].apply(max))
        batch_ids_dict["test"] = Dataset.from_dict({"start_id": begin_ids, "end_id": end_ids, "idx": list(range(len(begin_ids)))})
        batch_ids_dict["validation"] = Dataset.from_dict({})
        batch_ids_dict["train"] = Dataset.from_dict({})
        return DatasetDict(batch_ids_dict)

    for split in splits:
        dataset_pd = dataset[split].to_pandas()
        begin_ids = []
        end_ids = []
        print("Creating batch ids...")
        if "summary_id" in dataset[split].column_names:
            group_by_cols = ["doc_ids", "summary_id"]
        else:
            group_by_cols = ["doc_ids"]
        grouped = dataset_pd.groupby(group_by_cols)
        if "label" in dataset[split].column_names:
            # document level labels
            labels_unique = grouped["label"].apply(set)
            # all samples of the same document_title should have same label
            # (otherwise implement multi-label setting)
            if any(labels_unique.apply(len) > 1):
                raise ValueError(
                    "Encountered contradictive labels for the same document_title!"
                    "Please check data or implement multi-label setting."
                )
            else:
                tmp = (
                    grouped["label"]
                    .apply(list)
                    .reset_index(name="label")
                    .sort_values("doc_ids", ignore_index=True)
                )
                labels = list(tmp["label"].apply(lambda x: x[0]))
                tmp = (
                    grouped["document_title"]
                    .apply(list)
                    .reset_index(name="document_title")
                    .sort_values("doc_ids", ignore_index=True)
                )
                titles = list(tmp["document_title"].apply(lambda x: x[0]))
        tmp = grouped["sentence_ids"].apply(list).reset_index(name="sentence_ids")
        tmp = tmp.sort_values("doc_ids", ignore_index=True)
        begin_ids = list(tmp["sentence_ids"].apply(min))
        end_ids = list(tmp["sentence_ids"].apply(max))
        data_dict = {"start_id": begin_ids, "end_id": end_ids, "idx": list(range(len(begin_ids)))}
        if "label" in dataset[split].column_names:
            data_dict["label"] = labels
            data_dict["document_title"] = titles
        batch_ids_dict[split] = Dataset.from_dict(data_dict)
    return DatasetDict(batch_ids_dict)


def sent_files_exist(processed_data_path):
    okay = {}
    if os.path.exists(processed_data_path + "/all/sentences/"):
        return True
    for split in ["train", "validation", "test"]:
        if not os.path.exists(processed_data_path + f"/{split}/"):
            return False
        okay[split] = False
        if not os.path.exists(processed_data_path + f"/{split}/sentences/"):
            return False
        okay[split] = True
    return all(okay.values())

def merge_columns(examples, columns, new_column_name):
    new_column = " ".join([examples[col] for col in columns])
    examples[new_column_name] = new_column
    return examples