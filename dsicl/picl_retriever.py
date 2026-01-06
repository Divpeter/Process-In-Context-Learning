import numpy as np
import os
import re
from tqdm import trange, tqdm
import torch
from typing import List, Tuple, Dict, Union
from dsicl.retriever.base_retriever import BaseRetriever
from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter
from dsicl.data_reader import read_demo_benchmark
from FlagEmbedding import FlagReranker

class PiclRetriever(BaseRetriever):

    def __init__(self, 
                #  encoder: str,
                 prompter: Prompter) -> None:
        
        super(BaseRetriever, self).__init__()
        
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

        self.prompter = prompter

    def confusion_process(self, sample):
        #confusion summary extraction
        confusion_text = sample.outputs[0].text.rstrip()
        match = re.search(r'confusion{([^}]*)}', confusion_text)
        confusion = match.group(1) if match else ''

        return confusion

    def confusion_reranking(self, sample: Dict, confusion_text, recall_demos, shots: int) -> DataReader:
        #confusion aware reranking
        confusion = self.confusion_process(confusion_text)
        question_prompt = self.prompter.generate_prompt(sample)
        query_text = f"{question_prompt} {confusion}" if confusion else question_prompt

        candidates = [self.prompter.generate_prompt(d) for d in recall_demos]
        input_pairs = [[query_text, c] for c in candidates]

        scores = self.reranker.compute_score(input_pairs, normalize=True)

        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:shots]
        top_k_demos = [recall_demos[i] for i in top_k_idx][::-1]

        return DataReader(top_k_demos)


    def batch_confusion_reranking(self, data: DataReader, confusion_list, recall_demos, shots: int) -> List[DataReader]:
        return [
            self.confusion_reranking(d, c, r_demo, shots)
            for d, c, r_demo in tqdm(zip(data, confusion_list, recall_demos))
        ]

    