"""PICL inference with bounded confusion detection and dynamic insertion."""

from typing import Dict, List

from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter, extract_confusion
from dsicl.inferencer.base_inferencer import BaseInferencer
from dsicl.picl_retriever import PiclRetriever
from dsicl.utils import insertion_content


class PICLInferencer(BaseInferencer):
    def __init__(self, model, tokenizer, prompter: Prompter, labels, args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.labels = labels
        self.max_model_len = args.max_model_len
        self.max_tokens = args.max_tokens
        self.temperature = 0.7
        self.top_p = 0.8
        self.top_k_sampling = 20
        self.repetition_penalty = 1.05
        self.end_tokens = list(dict.fromkeys(
            token for token in ('<|im_end|>', self.tokenizer.eos_token) if token
        ))
        self.interrupt_set = (' Wait', ' wait')
        self.interrupt_tokens = [*self.interrupt_set, *self.end_tokens]
        self.r = args.r
        self.k = args.k
        self.retriever = None
        self.reranker_model_path = args.reranker_model_path
        self.m = args.m

    def _generate(self, contexts, detect_interrupts=False):
        if not contexts:
            return []
        from vllm import SamplingParams

        params = []
        for context in contexts:
            remaining = self.max_model_len - len(self.tokenizer.encode(context))
            if remaining < 1:
                raise ValueError(
                    'PICL context exceeds --max_model_len. Increase it, or reduce --k/--r/--max_tokens.'
                )
            params.append(SamplingParams(
                max_tokens=min(self.max_tokens, remaining),
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k_sampling,
                repetition_penalty=self.repetition_penalty,
                stop=self.interrupt_tokens if detect_interrupts else self.end_tokens,
                include_stop_str_in_output=True,
            ))
        return self.model.generate(contexts, sampling_params=params)

    def infer_reason_picl(self, demos, samples):
        unfinished = [
            {'index': idx, 'context': self.prompter.generate_context([], sample), 'question': sample}
            for idx, sample in enumerate(samples[:self.m])
        ]
        finished = []
        for _ in range(self.r):
            if not unfinished:
                break
            outputs = self._generate([item['context'] for item in unfinished], detect_interrupts=True)
            interrupted = []
            for item, output in zip(unfinished, outputs):
                text = output.outputs[0].text
                if text.rstrip().endswith(self.interrupt_set):
                    item['context'] += text
                    interrupted.append(item)
                else:
                    finished.append((item['index'], output))
            unfinished = interrupted
            if not unfinished:
                break

            detector_prompts = self.prompter.generate_confusion_query(
                [item['context'] for item in unfinished]
            )
            detections = self._generate(detector_prompts)
            confusion_indices = [
                idx for idx, detection in enumerate(detections)
                if extract_confusion(detection.outputs[0].text)
            ]
            if confusion_indices:
                if self.retriever is None:
                    self.retriever = PiclRetriever(self.prompter, self.reranker_model_path)
                selected = self.retriever.batch_confusion_reranking(
                    [unfinished[idx]['question'] for idx in confusion_indices],
                    [detections[idx] for idx in confusion_indices],
                    [demos[unfinished[idx]['index']] for idx in confusion_indices],
                    self.k,
                )
                for idx, demonstrations in zip(confusion_indices, selected):
                    if not demonstrations:
                        continue
                    demo_text = '\n\n'.join(
                        self.prompter.generate_label_prompt(demo) for demo in demonstrations
                    )
                    unfinished[idx]['context'] += insertion_content(demo_text)

        if unfinished:
            outputs = self._generate([item['context'] for item in unfinished])
            finished.extend((item['index'], output) for item, output in zip(unfinished, outputs))
        return [output for _, output in sorted(finished, key=lambda item: item[0])]

    def batch_infer_picl(self, demos_l: List[DataReader], data: List[Dict]):
        return [self.infer_reason_picl(demos_l, data)]
