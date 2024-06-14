import datasets
from src.inference_model import NextLevelBERT, NextLevelBERTConfig
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from torchmetrics.retrieval import RetrievalMRR, RetrievalMAP, RetrievalHitRate
from torchmetrics import MetricCollection
from copy import deepcopy
import sentence_transformers


"""
In this example, we use the NextLevelBERT model to embed chapters and summaries of books from the booksum dataset.
Our goal is to retrieve the respective summaries of each chapter and compute some retrieval metrics (MRR, MAP, HitRate) to 
measure how well we did. You can also use the SentenceTransformer model (all-MiniLM-L6-v2) for this task and compare the results.
Note that this is not the version of the booksum data that we use in the paper (which we cleaned ourselves) but another,
publicly-available cleaned version of the original dataset. The resulting metric values will therefore be different.
"""

def main(model_name="NextLevelBERT", chunking_strategy=256):
    # Load the dataset
    dataset = datasets.load_dataset("ubaada/booksum-complete-cleaned", "chapters")
    # preprocess the dataset
    summary_dataset, full_text_dataset = preprocess_data(dataset)    
    
    # choose between NextLevelBERT and the popular MiniLM-L6 sentence transformer
    if model_name == "NextLevelBERT":
        hub_model_name = f"aiintelligentsystems/nextlevelbert-{chunking_strategy}"
        config = NextLevelBERTConfig.from_pretrained(hub_model_name)
        model = NextLevelBERT.from_pretrained(hub_model_name, config=config)
    elif "SentenceTransformer":
        model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
    all_document_vectors = []
    all_summary_vectors = []
    model.eval()
    
    # embed chapters
    full_text_dataset = full_text_dataset.with_format("torch")
    dataloader = DataLoader(full_text_dataset, batch_size=256)
    print("Embedding chapters...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # encode data into document vectors
            if type(model).__name__ == "NextLevelBERT":
                document_vectors = model.encode(batch['chapter'], encoder_batch_size=1024)
                all_document_vectors.append(document_vectors.cpu())
            else:
                document_vectors = model.encode(batch['chapter'])
                all_document_vectors.append(torch.from_numpy(document_vectors))
    
    # embed summaries
    summary_dataset = summary_dataset.with_format("torch")
    dataloader = DataLoader(summary_dataset, batch_size=256)
    print("Embedding summaries...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # encode data into summary vectors
            if type(model).__name__ == "NextLevelBERT":
                summary_vectors = model.encode(batch['summary'], encoder_batch_size=1024)
                all_summary_vectors.append(summary_vectors.cpu())
            else:
                summary_vectors = model.encode(batch['summary'])
                all_summary_vectors.append(torch.from_numpy(summary_vectors))
    
    # evaluate the result
    document_vectors = torch.nn.functional.normalize(torch.cat(all_document_vectors))
    summary_vectors = torch.nn.functional.normalize(torch.cat(all_summary_vectors))
    summary_dataset = summary_dataset.add_column("summary_vectors", list(summary_vectors.numpy()))
    # for each document vector retrieve most similar summary vector with faiss
    summary_dataset.add_faiss_index(column='summary_vectors')
    scores, retrieved_examples = summary_dataset.get_nearest_examples_batch('summary_vectors', document_vectors.numpy(), k=10)
    # compute MRR
    metrics = MetricCollection(
            [RetrievalMRR(), RetrievalMAP(), RetrievalHitRate()],
        )
    targets = torch.tensor([[j == i for j in retrieved_examples[i]["id"]] for i in range(len(retrieved_examples))])
    query_indices = torch.arange(len(document_vectors)).unsqueeze(1).repeat((1, 10))
    scores = torch.from_numpy(np.array(scores))
    # convert distance scores to similarity scores
    scores = torch.ones_like(scores) - scores
    metrics = metrics(scores, targets, query_indices)
    print({metric_name: value.item() for metric_name, value in metrics.items()})
        
    
def preprocess_data(data):
    # preprocess data
    # unpack all summaries into their own examples
    # create a separate dataset for full-text chapters and for summaries
    summary_dataset = data
    summary_dataset = datasets.concatenate_datasets([summary_dataset["train"], summary_dataset["validation"], summary_dataset["test"]])
    summary_dataset = summary_dataset.add_column("id", range(len(summary_dataset)))
    
    def unpack_summaries(examples):
        ids = []
        texts = []
        summaries = []
        for i in range(len(examples["id"])):
            for summary in examples["summary"][i]:
                ids.append(examples["id"][i])
                texts.append(examples["text"][i])
                summaries.append(summary["text"])
        return {"id": ids, "chapter": texts, "summary": summaries}
    
    full_text_dataset = deepcopy(summary_dataset)
    full_text_dataset = full_text_dataset.remove_columns(set(full_text_dataset.column_names) - {"text", "id", "book_title"})
    summary_dataset = summary_dataset.map(unpack_summaries, num_proc=4, batched=True, remove_columns=summary_dataset.column_names)
    full_text_dataset = full_text_dataset.map(lambda x: {"chapter": re.sub("\n", " ", x["text"].strip())}, num_proc=4)
    summary_dataset = summary_dataset.map(lambda x: {"summary": re.sub("\n", "", x["summary"].strip())}, num_proc=4)
    summary_dataset = summary_dataset.filter(lambda x: len(x["summary"]) > 0)
    full_text_dataset = full_text_dataset.filter(lambda x: len(x["chapter"]) > 0)
    return summary_dataset, full_text_dataset


if __name__ == '__main__':
    chunking_strategy = 256 # choose from 0, 16, 32, 64, 128, 256, 512. In practice 256 works best. Choose 0 for sentence-level chunks.
    main("NextLevelBERT", chunking_strategy)