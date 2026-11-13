"""Confusion-aware cross-encoder reranking for PICL."""

from typing import Dict, List

import numpy as np
from tqdm import tqdm

from dsicl.retriever.base_retriever import BaseRetriever
from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter, extract_confusion


class PiclRetriever(BaseRetriever):
    def __init__(self, prompter: Prompter, model_path='BAAI/bge-reranker-v2-m3'):
        super().__init__()
        from FlagEmbedding import FlagReranker

        self.reranker = FlagReranker(model_path, use_fp16=True)
        self.prompter = prompter

    def confusion_process(self, sample):
        return extract_confusion(sample.outputs[0].text)

    def confusion_reranking(self, sample: Dict, confusion_text, recall_demos, shots: int) -> DataReader:
        confusion = self.confusion_process(confusion_text)
        if not confusion or not recall_demos:
            return DataReader([])
        question_prompt = self.prompter.generate_prompt(sample)
        separator = self.reranker.tokenizer.sep_token
        # Section 4.3: q [SEP] demonstration [SEP] confusion, using the model's separator.
        input_pairs = [
            [question_prompt, self.prompter.generate_label_prompt(demo) + separator + confusion]
            for demo in recall_demos
        ]
        scores = np.atleast_1d(self.reranker.compute_score(input_pairs, normalize=True))
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:shots]
        return DataReader([recall_demos[i] for i in top_k_idx])

    def batch_confusion_reranking(self, data, confusion_list, recall_demos, shots: int) -> List[DataReader]:
        return [
            self.confusion_reranking(sample, confusion, candidates, shots)
            for sample, confusion, candidates in tqdm(
                zip(data, confusion_list, recall_demos), total=len(data), desc='Reranking'
            )
        ]
