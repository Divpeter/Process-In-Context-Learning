import torch
import json
import os
import numpy as np
import argparse
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from vllm import LLM, SamplingParams
from datetime import datetime

import sys
sys.path.append('../../ICL4LRM/')

from dsicl.utils import set_seed, get_answer
from dsicl.data_reader import read_demo_benchmark
from dsicl.prompter import Prompter
from dsicl.template import DEMO_TEMPLATE, DEMO_HEAD, HEAD
from dsicl.inferencer.direct_inferencer import PICLInferencer
from dsicl.evaluator import Evaluator
import argparse

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    # parser.add_argument('--model_name',default='DeepSeek-R1-Distill-Qwen-7B', type=str, help='model name')
    parser.add_argument('--model_name',default='DeepSeek-R1-Distill-Llama-8B', type=str, help='model name')

    #demonstration,model and dataset path
    parser.add_argument('--saving_path',default='./output/mmlu_math/Similarity/seed=317', type=str, help='demonstrations directory')
    parser.add_argument('--model_path',default='/root/autodl-tmp/modelscope/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B', type=str, help='model path')
    parser.add_argument('--data_path',default='../DEmO_data', type=str, help='path of dataset')

    parser.add_argument('--task',default='mmlu_math', type=str, help='dataset name,[gsm8k, math500, mmlu_math, amc, aime, mmlu_pro]')
    parser.add_argument('--method',default='Similarity', type=str, help='dataset name,[Picl, Zero, Similarity]')
    #we provide the correct way to use picl here  please set this parameter with True
    # parser.add_argument('--picl',default= True, type=bool, help='whether to use picl or not')
    #two core parameters
    parser.add_argument('--shots',default='4', type=int, help='few_shots number')
    parser.add_argument('--r',default='1', type=int, help='max interventions')
    parser.add_argument('--k',default=1, type=int, help='insertion count')
    #you can control the dataset size used for reasoning 
    parser.add_argument('--m',default=100, type=int, help='dataset size')
    #inference parameters
    parser.add_argument('--max_model_len',default=20000, type=int, help='dataset size')
    parser.add_argument('--gpu_memory_utilization',default=0.83, type=float, help='insertion count')
    args = parser.parse_args()
    return args

def load_sample_demos_pairs(saving_path):
    with open(f'{saving_path}/demos.json', 'r') as f:
        sample_demos_pairs = json.load(f)
    return sample_demos_pairs

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=False, use_fast=False, padding_side="right")

    model = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)

    sample_demos_pairs = load_sample_demos_pairs(args.saving_path)
    
    samples = [sample_demos_pair['sample'] for sample_demos_pair in sample_demos_pairs ]

    demos_l = [sample_demos_pair['demos'] for sample_demos_pair in sample_demos_pairs ]

    if args.method != 'Zero':
        template, head = DEMO_TEMPLATE[args.task], DEMO_HEAD[args.task]
    else:
        template, head = DEMO_TEMPLATE[args.task], HEAD[args.task]
        
    # template, head = DEMO_TEMPLATE[args.task], HEAD[args.task]

    prompter = Prompter(template=template, head=head, sep='\n')

    labels = read_demo_benchmark(data_path=args.data_path, task=args.task)[0].data_info['label_space']

    picl_inferencer = PICLInferencer(model, tokenizer, prompter, labels, args)

    y_p = []

    if args.method == 'Zero' or args.method == 'Similarity':
        y_p = picl_inferencer.batch_infer(demos_l, samples)
        # y_p = vllm_inferencer.batch_infer_entropy(demos_l, samples, args.type, args.k, args.shot)
    elif args.method == 'Picl':    
        # y_p = vllm_inferencer.batch_infer_loop(demos_l, samples, args.type, args.k, args.shot)
        # y_p = vllm_inferencer.batch_infer_reranker(demos_l, samples, args.type, args.k, args.shot)
        y_p = picl_inferencer.batch_infer_picl(demos_l, samples)
    # y_p = picl_inferencer.batch_infer_picl(demos_l, samples)

    # preds = y_p[0][0]

    preds = y_p[0]

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S") 
    
    evaluator = Evaluator()

    #get the ground truth of questions
    y_t = get_answer(args.task, samples)

    if args.task == 'mmlu_math' or args.task == 'mmlu_pro':
        acc = evaluator.evaluate_selections(preds, y_t, args.task)
    else:
        acc, pred_answer, normalized_pred_answer, normalized_ground_truth = evaluator.evaluate_predictions(preds, y_t, args.task)

    print(f"ACC: {acc}")
    # print(f"Pred_answer: {pred_answer}")
    # print(f"Normalized_pred_answer: {normalized_pred_answer}")
    # print(f"Normalized_ground_truth: {normalized_ground_truth}")

        # 保存到 JSON 文件
    evaluation_data = {
        'Accuracy': acc,
        # 'Pred_answer': pred_answer,
        # 'Normalized_pred_answer': normalized_pred_answer,
        # 'Normalized_ground_truth': normalized_ground_truth
    }

    # 写入到文件
    with open(f'{args.saving_path}/evaluation_{args.model_name}_{formatted_time}.json', 'w') as f:
        json.dump(evaluation_data, f, indent=4)

if __name__ == '__main__':
    args = paras_args()
    main(args)
