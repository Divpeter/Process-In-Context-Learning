import numpy as np
import torch
from typing import List, Tuple, Dict
from tqdm import trange
from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter
from dsicl.inferencer.base_inferencer import BaseInferencer
from dsicl.model import speculative_decoder
from dsicl.picl_retriever import PiclRetriever
from vllm import LLM, SamplingParams
import re
import json
import time
from transformers import pipeline
from tqdm import tqdm

class DirectInferencer(BaseInferencer):

    def __init__(self, model, tokenizer, prompter: Prompter, labels) -> None:
        super(BaseInferencer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.decoder = speculative_decoder(self.model, self.tokenizer)
        self.prompter = prompter
        self.labels = labels

    def infer_classify(self, demos: DataReader, sample: Dict):
        context = self.prompter.generate_context(demos, sample)
        # print(context)
        label_idx = self.decoder.decode(context, self.labels).argmax()
        return self.labels[label_idx]
    
    def infer_reason(self, demos: DataReader, sample: Dict, k):
        context = self.prompter.generate_context(demos, sample)
        ans = []
        # print(context)
        for epoch in range(k):
            # messages = [
            #     {"role": "user", "content": context},
            # ]

            # pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            # generated_text = pipe(context, max_length=20000)
            #match = re.search(r'\\boxed{(\d+)}', generated_text[0]['generated_text'])

            # generated_text, token_count = self.decoder.decode_text(context, self.labels)
            #vllm
            generated_text, token_count = self.decoder.decode_text(context, self.labels)
            # match = re.search(r'\\boxed{(\d+)}', generated_text)
            # matches = re.findall(r'\\boxed\{(.*?)\}', generated_text)
            # final_answer = matches[-1]

            final_answer = 0
            check_label = 0
            # if match:
            #     final_answer = match.group(1)
            # else:
            #     check_label = 1
            res = {'answer':generated_text, 'final_answer':final_answer, 'check':check_label}

            #gsm8k
            # paired_ans = [{'question': sample['question'], 'label':sample['label'], 'answer':generated_text, 'final_answer':final_answer,'check':check_label }]
            #math500
            paired_ans = [{'question': sample['problem'], 'demo': demos, 'label':sample['answer'], 'answer':generated_text, 'final_answer':final_answer,'check':check_label, 'length': token_count, 'solution': sample['solution'], 'subject': sample['subject'], 'level': sample['level']}]
            try:
                with open("./output/math500/Zero/seed=317/confusion_preds_3.json", "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = []  

            existing_data.extend(paired_ans)
            with open("./output/math500/Zero/seed=317/confusion_preds_3.json", "w") as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False) 
            ans.append(res)
        return ans
    
    def infer_picl(self, demos: DataReader, sample: Dict, k, process):
        context = self.prompter.generate_context([], sample)

        context += process
        context += "Here are some examples. Focus on the underlying logic of the examples rather than getting too caught up in the specific content."
        for demo in demos:
            context += self.prompter.generate_label_prompt(demo)

        ans = []
        # print(context)
        for epoch in range(k):
        
            generated_text, token_count = self.decoder.decode_text(context, self.labels)
            # match = re.search(r'\\boxed{(\d+)}', generated_text)
            matches = re.findall(r'\\boxed\{(.*?)\}', generated_text)
            final_answer = matches[-1]

            final_answer = 0
            check_label = 0
            # if match:
            #     final_answer = match.group(1)
            # else:
            #     check_label = 1
            res = {'answer':generated_text, 'final_answer':final_answer, 'check':check_label}

            #gsm8k
            # paired_ans = [{'question': sample['question'], 'label':sample['label'], 'answer':generated_text, 'final_answer':final_answer,'check':check_label }]
            #math500
            paired_ans = [{'question': sample['problem'], 'demo': demos, 'label': sample['answer'], 'prefix': context, 'answer':generated_text, 'final_answer':final_answer,'check':check_label, 'length': token_count, 'solution': sample['solution'], 'subject': sample['subject'], 'level': sample['level']}]
            try:
                with open("./output/math500/Similarity/seed=317/picl_1/preds.json", "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = []  

            existing_data.extend(paired_ans)
            with open("./output/math500/Similarity/seed=317/picl_1/preds.json", "w") as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False) 
            ans.append(res)
        return ans
    
    def batch_infer(self, demos_l: List[DataReader] , data: DataReader, task_type, k, shot) -> List[str]:
        if task_type == 'classify':
            return [self.infer_classify(demos_l[_], data[_]) for _ in trange(len(data))]
        elif task_type == 'reason':
            return [self.infer_reason(demos_l[_][0:shot], data[_], k) for _ in trange(len(data))]
        
    def batch_picl_infer(self, demos_l: List[DataReader] , data: DataReader,  process, task_type, k, shot) -> List[str]:
        return [self.infer_picl(demos_l[_][0:shot], data[_], k, process[_]) for _ in trange(len(data))]

class VLLMInferencer(BaseInferencer):

    def __init__(self, model, tokenizer, prompter: Prompter, labels, args, judger = None) -> None:
        super(BaseInferencer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.labels = labels
        self.max_tokens = 20000
        self.temperature = 0.7
        self.top_p = 0.8
        self.top_k_sampling = 20
        self.repetition_penalty = 1.05
        self.end_token = ["<|end_in_context_learning|>", " <|im_end|>", "<|im_end|>", self.tokenizer.eos_token, "\n<|im_end|>", " \n<|im_end|>"]
        self.end_token_2 = ["<|end_in_context_learning|>", " Wait", " wait", self.tokenizer.eos_token, "<|im_end|>", " <|im_end|>", "\n<|im_end|>", " \n<|im_end|>"]
        self.end_token_entropy = [" Wait", " wait", " ?",  " maybe", " Maybe", self.tokenizer.eos_token, "<|im_end|>", " <|im_end|>", "\n<|im_end|>", " \n<|im_end|>"]
        self.end_markers = ["Wait", "wait", "?", "maybe", "Maybe"]
        self.check_number = args.shot
        self.judger = judger
        # self.retriever = PiclRetriever(args.embedding_model_path, prompter, args.data_path, args.task, args.seed, args.embedding_path)
        self.retriever = PiclRetriever( prompter, args.data_path, args.task, args.seed)
        self.m = args.m
        self.shots = args.shot
        self.task = args.task
        self.n = args.n
        self.r_shots = args.r_shots

    def infer_reason(self, demos: DataReader, sample: Dict, k):
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
        start_time = time.time()
        total_number = self.m * self.n
        output_list = self.model.generate(context[0:total_number], sampling_params=sampling_params)
        end_time = time.time()
        execution_time = end_time - start_time
        return output_list, execution_time, context

    def infer_reason_loop(self, demos: DataReader, sample: Dict, k):
        context_with_index = [
            # {"index": idx, "context": self.prompter.generate_context(demos[idx][0:k], sample[idx])}
            {"index": idx, "context": self.prompter.generate_context([], sample[idx])}
            for idx in trange(len(sample[0:200]))
        ]

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k_sampling,
            repetition_penalty=self.repetition_penalty,
            stop=self.end_token_2,
            include_stop_str_in_output=True,
        )

        sampling_params_detect = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k_sampling,
            repetition_penalty=self.repetition_penalty,
            stop=self.end_token,
            include_stop_str_in_output=True,
        )

        loop = 0
        finished = []
        unfinished = context_with_index  # 待处理队列

        start_time = time.time()
        while loop < self.check_number and unfinished:
            # 当前轮生成
            current_contexts = [item["context"] for item in unfinished]
            output_list = self.model.generate(current_contexts, sampling_params=sampling_params)

            next_unfinished = []
            for item, output, context in zip(unfinished, output_list, current_contexts):
                output_text = output.outputs[0].text.rstrip()
                if output_text.endswith(tuple(self.end_markers)):
                    next_unfinished.append({
                        "index": item["index"],
                        "context": context + output_text

                    })
                else:
                    finished.append({
                        "index": item["index"],
                        "output": output,
                        "loop": loop,
                        "context": context
                    })

            print(f"{len(next_unfinished)} remained to be confusion_detect")
            unfinished = next_unfinished
            loop += 1

            # 对 unfinished 的 context 做 confusion detect 或修改
            context_confusion = self.prompter.generate_confusion_query([item["context"] for item in unfinished])

            #decide to use what judger
            if not self.judger:
                detect_list = self.model.generate(context_confusion, sampling_params=sampling_params_detect)
            else:
                detect_list = self.judger.generate(context_confusion, sampling_params=sampling_params_detect)
            
            for idx, detect in enumerate(detect_list):
                if 'yes' in detect.outputs[0].text.lower():
                    unfinished[idx]['context'] += f"Here is an example that might help clarify your confusion. Please focus on the logic rather than the content. Demo: {demos[unfinished[idx]['index']][loop]}. Remember the answer should be put in \\boxed"
            
            # for idx in range(len(unfinished)):
            #     unfinished[idx]['context'] += f"Here is an example that might help clarify your confusion. Please focus on the logic rather than the content. Demo: {demos[unfinished[idx]['index']][loop]}. Remember the answer should be put in \\boxed"


        # 最后一轮强制生成
        if unfinished:
            final_contexts = [item["context"] for item in unfinished]
            final_outputs = self.model.generate(final_contexts, sampling_params=sampling_params_detect)
            for item, output, context in zip(unfinished, final_outputs, final_contexts):
                finished.append({
                    "index": item["index"],
                    "output": output,
                    "loop": loop,
                    "context": context
                })

        # 按 index 排序返回
        finished_sorted = sorted(finished, key=lambda x: x["index"])
        end_time = time.time()
        execution_time = end_time - start_time
        return [item["output"] for item in finished_sorted], execution_time, [item["loop"] for item in finished_sorted], [item["context"] for item in finished_sorted]

    def infer_reason_reranker(self, demos: DataReader, sample: Dict, k):
        total_number = self.m * self.n
        context_with_index = [
            # {"index": idx, "context": self.prompter.generate_context(demos[idx][0:k], sample[idx])}
            {"index": idx, "context": self.prompter.generate_context([], sample[idx]), "confusion_detect":[], "question": sample[idx], "demos": [] }
            for idx in trange(len(sample[0:total_number]))
        ]

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k_sampling,
            repetition_penalty=self.repetition_penalty,
            stop=self.end_token_2,
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

        loop = 0
        finished = []
        unfinished = context_with_index  # 待处理队列

        start_time = time.time()
        while loop < self.check_number and unfinished:
            # 当前轮生成
            current_contexts = [item["context"] for item in unfinished]
            output_list = self.model.generate(current_contexts, sampling_params=sampling_params)

            next_unfinished = []
            for item, output, context in zip(unfinished, output_list, current_contexts):
                output_text = output.outputs[0].text.rstrip()
                if output_text.endswith(tuple(self.end_markers)):
                    next_unfinished.append({
                        "index": item["index"],
                        "context": context + output_text,
                        "confusion_detect": item["confusion_detect"],
                        "question": item['question'],
                        "demos": item["demos"]
                    })
                else:
                    finished.append({
                        "index": item["index"],
                        "output": output,
                        "loop": loop,
                        "context": context,
                        "confusion_detect": item["confusion_detect"],
                        "question": item['question'],
                        "demos": item["demos"]
                    })

            print(f"{len(next_unfinished)} remained to be confusion_detect")
            unfinished = next_unfinished
            loop += 1

            # 对 unfinished 的 context 做 confusion detect 或修改
            context_confusion = self.prompter.generate_confusion_query([item["context"] for item in unfinished])

            #decide to use what judger
            if not self.judger:
                detect_list = self.model.generate(context_confusion, sampling_params=sampling_params_detect)
            else:
                detect_list = self.judger.generate(context_confusion, sampling_params=sampling_params_detect)
            
            # for idx, detect in enumerate(detect_list):
            #     if 'yes' in detect.outputs[0].text.lower():
            #         unfinished[idx]['context'] += f"Here is an example that might help clarify your confusion. Please focus on the logic rather than the content. Demo: {demos[unfinished[idx]['index']][loop]}. Remember the answer should be put in \\boxed"
            
            # for idx in range(len(unfinished)):
            #     unfinished[idx]['context'] += f"Here is an example that might help clarify your confusion. Please focus on the logic rather than the content. Demo: {demos[unfinished[idx]['index']][loop]}. Remember the answer should be put in \\boxed"


            confusion_idx = []
            # for idx, detect in enumerate(detect_list):

            #     unfinished[idx]['confusion_detect'].append(detect.outputs[0].text.rstrip())

            #     confusion_pattern = r"confusion\{.*?\}"

            #     # if 'yes' in detect.outputs[0].text.lower() or re.search(confusion_pattern, detect.outputs[0].text.lower()):
            #     #     confusion_idx.append(idx)
            #      # Search for the word 'yes' and the non-empty confusion pattern
            #     if 'yes' in detect.outputs[0].text.lower():
            #         confusion_idx.append(idx)
            #     else:
            #         match = re.search(confusion_pattern, detect.outputs[0].text.lower())
            #         if match and match.group(1).strip():  # Ensure the matched content is not empty
            #             confusion_idx.append(idx)
            #         # unfinished[idx]['context'] += f"Here is an example that might help clarify your confusion. Please focus on the logic rather than the content. Demo: {demos[unfinished[idx]['index']][loop]}. Remember the answer should be put in \\boxed"
            
            for idx, detect in enumerate(detect_list):
                text = detect.outputs[0].text.rstrip()
                unfinished[idx]['confusion_detect'].append(text)

                confusion_pattern = r"confusion\{(.*?)\}"
                match = re.search(confusion_pattern, text.lower())

                if 'yes' in text.lower():
                    confusion_idx.append(idx)
                elif match:
                    inner_text = match.group(1).strip()
                    if inner_text:  # Only append if the content inside braces is not empty
                        confusion_idx.append(idx)
                        
            confusion_detect_list = [detect_list[idx] for idx in confusion_idx]
            question_list = [unfinished[idx]['question'] for idx in confusion_idx]

            recall_demos  = [demos[unfinished[idx]['index']] for idx in confusion_idx]

            # demos = self.retriever.weighted_batch_retrieve(question_list, confusion_detect_list, 10)
            rerank_demos = self.retriever.weighted_batch_retrieve_bge(question_list, confusion_detect_list, recall_demos, self.r_shots)
            # if self.LLM_use == True:
            #     context_confusion = self.prompter.generate_confusion_query([item["context"] for item in unfinished])

            for idx, ele in enumerate(confusion_idx):
                demo_r = ''
                for r_idx in range(self.r_shots):
                    if self.task == 'gsm8k':
                        demo_r += f"question:{rerank_demos[idx].data[r_idx]['question']} answer:{rerank_demos[idx].data[r_idx]['answer']} "
                    elif self.task == 'math500':
                        demo_r += f"question:{rerank_demos[idx].data[r_idx]['problem']} answer:{rerank_demos[idx].data[r_idx]['solution']} "
                    elif self.task == 'aime':
                        demo_r += f"question:{rerank_demos[idx].data[r_idx]['Problem']} answer:{rerank_demos[idx].data[r_idx]['Solution']} "
                    elif self.task == 'sat_math':
                        demo_r += f"question:{rerank_demos[idx].data[r_idx]['question']} choices:{rerank_demos[idx].data[r_idx]['choices']} answer:{rerank_demos[idx].data[r_idx]['answer'] } "
                    elif self.task == 'mmlu_pro' or self.task == 'mmlu_math' :
                        demo_r += f"question:{rerank_demos[idx].data[r_idx]['question']} choices:{rerank_demos[idx].data[r_idx]['choices']} answer:{rerank_demos[idx].data[r_idx]['answer'] } "
                    elif self.task == 'amc':
                        demo_r += f"question:{rerank_demos[idx].data[r_idx]['question']} answer:{rerank_demos[idx].data[r_idx]['solution']} "
                    
                    unfinished[ele]['demos'].append(rerank_demos[idx].data[r_idx])
                unfinished[ele]['context'] += f"Here is an example that might help clarify your confusion. Demo: {demo_r}. Please focus on the logic rather than the content of the provided demo. Remember the answer should be put in \\boxed{{}}"
                # if self.task == 'gsm8k':
                #     unfinished[ele]['context'] += f"Here is an example that might help clarify your confusion. Demo: question:{rerank_demos[idx].data[0]['question']} answer:{rerank_demos[idx].data[0]['answer']}. Please focus on the logic rather than the content of the provided demo. Remember the answer should be put in \\boxed{{}}"
                # elif self.task == 'math500':
                #     unfinished[ele]['context'] += f"Here is an example that might help clarify your confusion. Demo: question:{rerank_demos[idx].data[0]['problem']} answer:{rerank_demos[idx].data[0]['solution']}. Please focus on the logic rather than the content of the provided demo. Remember the answer should be put in \\boxed{{}}"
                # elif self.task == 'aime':
                #     unfinished[ele]['context'] += f"Here is an example that might help clarify your confusion. Demo: question:{rerank_demos[idx].data[0]['Problem']} answer:{rerank_demos[idx].data[0]['Solution']}. Please focus on the logic rather than the content of the provided demo. Remember the answer should be put in \\boxed{{}}"
                # elif self.task == 'sat_math':
                #     unfinished[ele]['context'] += f"Here is an example that might help clarify your confusion. Demo: question:{rerank_demos[idx].data[0]['question']} choices:{rerank_demos[idx].data[0]['choices']} answer:{rerank_demos[idx].data[0]['answer']}. Please focus on the logic rather than the content of the provided demo. Remember the answer should be put in \\boxed{{}}"
                # elif self.task == 'mmlu_pro':
                #     unfinished[ele]['context'] += f"Here is an example that might help clarify your confusion. Demo: question:{rerank_demos[idx].data[0]['question']} choices:{rerank_demos[idx].data[0]['choices']} answer:{rerank_demos[idx].data[0]['answer']}. Please focus on the logic rather than the content of the provided demo. Remember the answer should be put in \\boxed{{}}"
                # elif self.task == 'amc':
                #     unfinished[ele]['context'] += f"Here is an example that might help clarify your confusion. Demo: question:{rerank_demos[idx].data[0]['question']} answer:{rerank_demos[idx].data[0]['solution']}. Please focus on the logic rather than the content of the provided demo. Remember the answer should be put in \\boxed{{}}"
               
                
            # for idx in range(len(unfinished)):
            #     unfinished[idx]['context'] += f"Here is an example that might help clarify your confusion. Please focus on the logic rather than the content. Demo: {demos[idx][loop]}. Remember the answer should be put in \\boxed"

        # 最后一轮强制生成
        if unfinished:
            final_contexts = [item["context"] for item in unfinished]
            final_outputs = self.model.generate(final_contexts, sampling_params=sampling_params_detect)
            for item, output, context in zip(unfinished, final_outputs, final_contexts):
                finished.append({
                    "index": item["index"],
                    "output": output,
                    "loop": loop,
                    "context": context,
                    "confusion_detect": item["confusion_detect"],
                    "question": item['question'],
                    "demos": item["demos"]
                })

        # 按 index 排序返回

        
        finished_sorted = sorted(finished, key=lambda x: x["index"])
        end_time = time.time()
        execution_time = end_time - start_time
        return [item["output"] for item in finished_sorted], execution_time, [item["loop"] for item in finished_sorted], [item["context"] for item in finished_sorted], [item["confusion_detect"] for item in finished_sorted], [item["demos"] for item in finished_sorted]


    
    def infer_picl(self, demos: DataReader, sample: Dict, k, process):
        context = self.prompter.generate_context([], sample)

        context += process
        context += "Here are some examples. Focus on the underlying logic of the examples rather than getting too caught up in the specific content."
        for demo in demos:
            context += self.prompter.generate_label_prompt(demo)

        ans = []
        # print(context)
        for epoch in range(k):
        
            generated_text, token_count = self.decoder.decode_text(context, self.labels)
            # match = re.search(r'\\boxed{(\d+)}', generated_text)
            matches = re.findall(r'\\boxed\{(.*?)\}', generated_text)
            final_answer = matches[-1]

            final_answer = 0
            check_label = 0
            # if match:
            #     final_answer = match.group(1)
            # else:
            #     check_label = 1
            res = {'answer':generated_text, 'final_answer':final_answer, 'check':check_label}

            #gsm8k
            # paired_ans = [{'question': sample['question'], 'label':sample['label'], 'answer':generated_text, 'final_answer':final_answer,'check':check_label }]
            #math500
            paired_ans = [{'question': sample['problem'], 'demo': demos, 'label': sample['answer'], 'prefix': context, 'answer':generated_text, 'final_answer':final_answer,'check':check_label, 'length': token_count, 'solution': sample['solution'], 'subject': sample['subject'], 'level': sample['level']}]
            try:
                with open("./output/math500/Similarity/seed=317/picl_1/preds.json", "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = []  

            existing_data.extend(paired_ans)
            with open("./output/math500/Similarity/seed=317/picl_1/preds.json", "w") as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False) 
            ans.append(res)
        return ans
    
    def infer_reason_entropy(self, demos: DataReader, sample: Dict, k):
        context = [self.prompter.generate_context(demos[_][0:0], sample[_]) for _ in trange(len(sample[0:1300]))]
        
        # with open('/root/ICL4LRM/DEmO/output/gsm8k/Similarity/seed=317/preds_50_manual_check copy.json', "r") as f:
        #     res = json.load(f)
        # picl_label = 'an example that might help clarify your confusion'
        # for idx, ele in enumerate(res):
        #     if picl_label in ele['context']:
        #         context[idx] = ele['context']
        #         print(idx)

        sampling_params = SamplingParams(
            max_tokens= self.max_tokens,
            temperature= self.temperature,
            top_p= self.top_p,
            top_k= self.top_k_sampling,
            repetition_penalty= self.repetition_penalty,
            stop=[self.tokenizer.eos_token],
            # stop=[self.end_token, self.tokenizer.eos_token],
            include_stop_str_in_output=True,
            logprobs=20
        )
        
        start_time = time.time()
        output_list = self.model.generate(context[0:1300], sampling_params=sampling_params)

        results_json = []

        for output in tqdm(output_list, desc="Calculating Entropy"):
            ele_res = []
            for i, prob in enumerate(output.outputs[0].logprobs):
                logprob_values = np.array([v.logprob for v in prob.values()])
                probs = np.exp(logprob_values)
                probs /= probs.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-12))  # 防止 log(0)
                     
                token_id = output.outputs[0].token_ids[i]
                token_text = self.tokenizer.decode([token_id])
                # logprob_info = output.logprobs[i]

                token_entry = {
                    "token_index": i,
                    "token_id": token_id,
                    "token_text": token_text,
                    'entropy': entropy
                    # "logprobs": logprob_info,  # 原始字典结构
                }

                ele_res.append(token_entry)
            results_json.append(ele_res)     

        with open("/root/ICL4LRM/DEmO/output/gsm8k/Entropy/entropy_1300.json", 'w') as f:
            json.dump(results_json, f)
        return  results_json
    
    def batch_infer(self, demos_l: List[DataReader] , data: DataReader, task_type, k, shot) -> List[str]:
        if task_type == 'classify':
            return None
        elif task_type == 'reason':
            # return [self.infer_reason(demos_l[_][0:shot], data[_], k) for _ in trange(len(data))]\
            return [self.infer_reason(demos_l, data, k)]

    def batch_infer_entropy(self, demos_l: List[DataReader] , data: DataReader, task_type, k, shot) -> List[str]:
        if task_type == 'classify':
            return None
        elif task_type == 'reason':
            # return [self.infer_reason(demos_l[_][0:shot], data[_], k) for _ in trange(len(data))]\
            return [self.infer_reason_entropy(demos_l, data, k)]   

        
    def batch_infer_loop(self, demos_l: List[DataReader] , data: DataReader, task_type, k, shot) -> List[str]:
        if task_type == 'classify':
            return None
        elif task_type == 'reason':
            # return [self.infer_reason(demos_l[_][0:shot], data[_], k) for _ in trange(len(data))]\
            return [self.infer_reason_loop(demos_l, data, k)]
        
            
    def batch_infer_reranker(self, demos_l: List[DataReader] , data: DataReader, task_type, k, shot) -> List[str]:
        if task_type == 'classify':
            return None
        elif task_type == 'reason':
            # return [self.infer_reason(demos_l[_][0:shot], data[_], k) for _ in trange(len(data))]\
            return [self.infer_reason_reranker(demos_l, data, k)]
        
    def batch_picl_infer(self, demos_l: List[DataReader] , data: DataReader,  process, task_type, k, shot) -> List[str]:
        # return [self.infer_picl(demos_l[_][0:shot], data[_], k, process[_]) for _ in trange(len(data))]
        return [self.infer_picl(demos_l, data, k, process) ]
