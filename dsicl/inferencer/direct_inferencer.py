from typing import List, Dict
from tqdm import trange
from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter
from dsicl.inferencer.base_inferencer import BaseInferencer
from dsicl.picl_retriever import PiclRetriever
from dsicl.utils import insertion_content
from vllm import SamplingParams
import re

class PICLInferencer(BaseInferencer):

    def __init__(self, model, tokenizer, prompter: Prompter, labels, args) -> None:
        super(BaseInferencer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.labels = labels

        #LLM inference parameters
        self.max_tokens = args.max_model_len
        self.temperature = 0.7
        self.top_p = 0.8
        self.top_k_sampling = 20
        self.repetition_penalty = 1.05
        self.end_token = [" <|im_end|>", "<|im_end|>", self.tokenizer.eos_token, "\n<|im_end|>", " \n<|im_end|>"]
        self.interrupt_tokens = [" Wait", " wait", self.tokenizer.eos_token, "<|im_end|>", " <|im_end|>", "\n<|im_end|>", " \n<|im_end|>"]
        self.interrupt_set = [" Wait", " wait"]
        self.r = args.r
        self.k = args.k
        self.shots = args.shots
        self.retriever = PiclRetriever(prompter)
        self.m = args.m
        self.task = args.task
    
    def infer_reason(self, demos: DataReader, sample: Dict):
        # context = [self.prompter.generate_context(demos[_][0:4], sample[_]) for _ in trange(len(sample))]
        context = [self.prompter.generate_context(demos[_][0:self.shots], sample[_]) for _ in trange(len(sample))]
        sampling_params = SamplingParams(
            max_tokens= self.max_tokens,
            temperature= self.temperature,
            top_p= self.top_p,
            top_k= self.top_k_sampling,
            repetition_penalty= self.repetition_penalty,
            # stop=[self.tokenizer.eos_token],
            stop=self.end_token,
            # stop=[self.end_token, self.tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        output_list = self.model.generate(context[0:self.m], sampling_params=sampling_params)
        return output_list

    def infer_reason_picl(self, demos: DataReader, sample: Dict):
        context_with_index = [
            {"index": idx, "context": self.prompter.generate_context([], sample[idx]), "question": sample[idx] }
            for idx in trange(len(sample[0:self.m]))
        ]
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k_sampling,
            repetition_penalty=self.repetition_penalty,
            stop=self.interrupt_tokens,#interrupt mechanism
            include_stop_str_in_output=True,
        )

        sampling_params_detect = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k_sampling,
            repetition_penalty=self.repetition_penalty,
            stop=self.tokenizer.eos_token,
            include_stop_str_in_output=True,
        )

        intervention_cnt = 0
        finished = [] # finished question list 
        unfinished = context_with_index  # question list remained to be process

        while intervention_cnt < self.r and unfinished:
            current_contexts = [item["context"] for item in unfinished]
            output_list = self.model.generate(current_contexts, sampling_params=sampling_params)
            next_unfinished = []
            for item, output, context in zip(unfinished, output_list, current_contexts):
                output_text = output.outputs[0].text.rstrip()
                if output_text.endswith(tuple(self.interrupt_set)):
                    next_unfinished.append({
                        "index": item["index"],
                        "context": context + output_text,
                        "question": item['question']
                    })
                else:
                    finished.append({
                        "index": item["index"],
                        "output": output,
                        "context": context,
                        "question": item['question']
                    })

            print(f"{len(next_unfinished)} remained to be confusion_detect")
            unfinished = next_unfinished
            intervention_cnt += 1

            # confusion analysis 
            context_confusion = self.prompter.generate_confusion_query([item["context"] for item in unfinished])
            detect_list = self.model.generate(context_confusion, sampling_params=sampling_params_detect)
            
            confusion_idx = []
            for idx, detect in enumerate(detect_list):
                text = detect.outputs[0].text.rstrip()

                confusion_pattern = r"confusion\{(.*?)\}"
                match = re.search(confusion_pattern, text.lower())

                #confuson identification
                if 'yes' in text.lower():
                    confusion_idx.append(idx)
                elif match:
                    inner_text = match.group(1).strip()
                    if inner_text:  # Only append if the content inside braces is not empty
                        confusion_idx.append(idx)
                        
            confusion_detect_list = [detect_list[idx] for idx in confusion_idx]
            question_list = [unfinished[idx]['question'] for idx in confusion_idx]
            
            #recall demos from semantic retrieval
            recall_demos  = [demos[unfinished[idx]['index']] for idx in confusion_idx]
            
            #reranking
            rerank_demos = self.retriever.batch_confusion_reranking(question_list, confusion_detect_list, recall_demos, self.k)

            #demonstrations insertion
            for idx, ele in enumerate(confusion_idx):
                demo_k = ''
                for k_idx in range(self.k):
                    if self.task == 'gsm8k':
                        demo_k += f"question:{rerank_demos[idx].data[k_idx]['question']} answer:{rerank_demos[idx].data[k_idx]['answer']} "
                    elif self.task == 'math500':
                        demo_k += f"question:{rerank_demos[idx].data[k_idx]['problem']} answer:{rerank_demos[idx].data[k_idx]['solution']} "
                    elif self.task == 'aime':
                        demo_k += f"question:{rerank_demos[idx].data[k_idx]['Problem']} answer:{rerank_demos[idx].data[k_idx]['Solution']} "
                    elif self.task == 'sat_math':
                        demo_k += f"question:{rerank_demos[idx].data[k_idx]['question']} choices:{rerank_demos[idx].data[k_idx]['choices']} answer:{rerank_demos[idx].data[k_idx]['answer'] } "
                    elif self.task == 'mmlu_pro' or self.task == 'mmlu_math' :
                        demo_k += f"question:{rerank_demos[idx].data[k_idx]['question']} choices:{rerank_demos[idx].data[k_idx]['choices']} answer:{rerank_demos[idx].data[k_idx]['answer'] } "
                    elif self.task == 'amc':
                        demo_k += f"question:{rerank_demos[idx].data[k_idx]['question']} answer:{rerank_demos[idx].data[k_idx]['solution']} "
                unfinished[ele]['context'] += insertion_content(demo_k)
       
        if unfinished:
            final_contexts = [item["context"] for item in unfinished]
            final_outputs = self.model.generate(final_contexts, sampling_params=sampling_params_detect)
            for item, output, context in zip(unfinished, final_outputs, final_contexts):
                finished.append({
                    "index": item["index"],
                    "output": output,
                    "context": context,
                    "question": item['question'],
                })

        finished_sorted = sorted(finished, key=lambda x: x["index"])
        return [item["output"] for item in finished_sorted]
            
    def batch_infer_picl(self, demos_l: List[DataReader] , data: DataReader) -> List[str]:
        return [self.infer_reason_picl(demos_l, data)]
    
    def batch_infer(self, demos_l: List[DataReader] , data: DataReader) -> List[str]:
        return [self.infer_reason(demos_l, data)]
        
