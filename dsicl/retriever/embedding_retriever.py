import numpy as np
import os
from tqdm import trange, tqdm
import torch
from typing import List, Tuple, Dict, Union
from dsicl.retriever.base_retriever import BaseRetriever
from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter

class EmbeddingRetriever(BaseRetriever):

    def __init__(self,
                 encoder: str,
                 prompter: Prompter,
                 data: DataReader = None,
                 embedding_path: str = None) -> None:

        super(BaseRetriever, self).__init__()

        from dsicl.model import get_encoder

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = get_encoder(encoder, device=device)
        self.prompter = prompter
        self.similarity_metric = 'l2'

        if not data is None:
            self.data = data
            self.embeddings = None

        if not embedding_path is None:
            self.embeddings = torch.load(embedding_path)


    def encode(self, x: str) -> torch.FloatTensor:
        return self.encoder(x)

    def save_embeddings(self, path):
        # torch.save(self.embeddings, f"embedding/{self.paras['task']}_{self.paras['encoder']}.pt")
        torch.save(self.embeddings, path)

    def build_embeddings(self) -> None:
        embeddings = []
        for i in trange(len(self.data)):
            x = self.prompter.generate_prompt(self.data[i])
            v = self.encode(x)
            embeddings.append(v)
        self.embeddings = torch.stack(embeddings)
#        torch.save(embeddings, f"embedding/{self.paras['task']}_{self.paras['encoder']}.pt")

    def retrieve(self, sample: Dict, shots: int, exclude_query: bool = False) -> DataReader:
        q = self.encode(self.prompter.generate_prompt(sample))
        if self.similarity_metric == 'l2':
            distances = torch.sum((q - self.embeddings) ** 2, dim=1) ** 0.5
            idxs = distances.argsort().tolist()
        elif self.similarity_metric == 'cosine':
            scores = torch.nn.functional.cosine_similarity(q.unsqueeze(0), self.embeddings, dim=1)
            idxs = scores.argsort(descending=True).tolist()
        else:
            raise ValueError("similarity_metric must be 'l2' or 'cosine'")
        if exclude_query:
            query = self.prompter.generate_prompt(sample)
            idxs = [idx for idx in idxs if self.prompter.generate_prompt(self.data[idx]) != query]
        return DataReader([self.data[idx] for idx in idxs[:shots]])

    def batch_retrieve(self, data: DataReader, shots: int) -> List[DataReader]:

        return [self.retrieve(d, shots) for d in tqdm(data)]
