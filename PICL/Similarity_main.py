import torch
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('../../ICL4LRM/')

from dsicl.utils import set_seed
from dsicl.data_reader import DataReader, read_demo_benchmark
from dsicl.prompter import Prompter
from dsicl.template import DEMO_TEMPLATE, DEMO_HEAD
from dsicl.retriever import EmbeddingRetriever

import argparse


def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/mmlu_math/Similarity/seed=317', type=str, help='Directory to save the demonstrations')
    parser.add_argument('--data_path',default='../DEmO_data', type=str, help='Path of DEmO data')
    parser.add_argument('--task',default='mmlu_math', type=str, help='dataset name')
    parser.add_argument('--shots',default = 4, type=int, help='Number of demonstrations')
    parser.add_argument('--embedding_model_path',default='/root/autodl-tmp/modelscope/models/AI-ModelScope/bert-base-uncased', type=str, help='The model to use for embedding')
    parser.add_argument('--type',default='reason', type=str, help='Type of task')
    args = parser.parse_args()
    return args

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.saving_path):
        os.makedirs(args.saving_path)

    pool, testset = read_demo_benchmark(data_path=args.data_path, task=args.task)

    template, head = DEMO_TEMPLATE[args.task], DEMO_HEAD[args.task]

    prompter = Prompter(template=template, head=head, sep='\n')

    # set_seed(args.seed)
    
    if args.type == 'reason':
        original_demos = pool.get_subset(len(pool), balance = False)
    else:
        original_demos = pool.get_subset(len(pool.data_info['label_space'])*args.shots, balance=True)

    retriver = EmbeddingRetriever(encoder=args.embedding_model_path, prompter=prompter, data=original_demos)
    
    retriver.build_embeddings()

    #semantic_retrieval
    if args.type == 'reason':
        demos_l = [retriver.retrieve(d, args.shots).to_list() for d in tqdm(testset)]
    else:
        demos_l = [retriver.retrieve(d, len(original_demos)).to_list() for d in tqdm(testset)]

    filtered_sample_demos_pairs = []
    for i in range(len(testset)):
        filtered_demos = [demo for demo in demos_l[i] if demo != testset[i]]
        filtered_sample_demos_pairs.append({'sample': testset[i], 'demos': filtered_demos})

    with open(f'{args.saving_path}/demos.json', 'w') as f:
        json.dump(filtered_sample_demos_pairs, f, indent=4)

    # Save arguments
    with open(f'{args.saving_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    args = paras_args()
    main(args)
