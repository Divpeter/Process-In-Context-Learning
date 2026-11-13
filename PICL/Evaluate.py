"""Run PICL inference and evaluation from a semantic candidate cache."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def paras_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model_name', help='Optional name used in result filenames')
    parser.add_argument('--saving_path', required=True, help='Directory containing demos.json')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--reranker_model_path', default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--task', default='gsm8k')
    parser.add_argument('--r', type=int, default=1, help='Maximum interventions per question')
    parser.add_argument('--k', type=int, default=1, help='Demonstrations per intervention')
    parser.add_argument('--m', type=int, help='Evaluate only the first m questions; default: all')
    parser.add_argument('--max_model_len', type=int, default=20000)
    parser.add_argument('--max_tokens', type=int, default=20000, help='Maximum new tokens per call')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.83)
    parser.add_argument('--seed', type=int, default=317)
    args = parser.parse_args(argv)
    if args.r < 1 or args.k < 1 or (args.m is not None and args.m < 1):
        parser.error('--r, --k, and --m must be positive')
    if args.max_model_len < 2 or args.max_tokens < 1:
        parser.error('--max_model_len must be at least 2 and --max_tokens must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('--gpu_memory_utilization must be between 0 and 1')
    return args


def load_sample_demos_pairs(saving_path, task=None):
    saving_path = Path(saving_path)
    metadata_path = saving_path / 'retrieval_args.json'
    if task is not None and metadata_path.exists():
        with metadata_path.open(encoding='utf-8') as handle:
            cached_task = json.load(handle)['task']
        if cached_task != task:
            raise ValueError(f'Candidate cache is for {cached_task}, not {task}. Rerun retrieval.')
    with (saving_path / 'demos.json').open(encoding='utf-8') as handle:
        pairs = json.load(handle)
    if not pairs:
        raise ValueError('demos.json contains no questions.')
    return pairs


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pairs = load_sample_demos_pairs(args.saving_path, args.task)[:args.m]
    samples = [pair['sample'] for pair in pairs]
    demos_l = [pair['demos'] for pair in pairs]

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM
    from dsicl.utils import set_seed, get_answer
    from dsicl.prompter import Prompter
    from dsicl.template import DEMO_TEMPLATE, PICL_HEAD
    from dsicl.inferencer.direct_inferencer import PICLInferencer
    from dsicl.evaluator import Evaluator

    if not torch.cuda.is_available():
        raise RuntimeError('PICL inference requires a CUDA GPU and vLLM.')
    set_seed(args.seed)
    prompter = Prompter(DEMO_TEMPLATE[args.task], PICL_HEAD[args.task], task=args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='right')
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    inferencer = PICLInferencer(model, tokenizer, prompter, [], args)
    preds = inferencer.batch_infer_picl(demos_l, samples)[0]
    evaluator = Evaluator()
    y_t = get_answer(args.task, samples)
    if args.task in ('mmlu_math', 'mmlu_pro'):
        acc = evaluator.evaluate_selections(preds, y_t)
    else:
        acc, _, _, _ = evaluator.evaluate_predictions(preds, y_t, args.task)
    print(f'Accuracy: {acc}')

    model_name = args.model_name or Path(args.model_path.rstrip('/\\')).name
    model_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', model_name)
    run_name = f'{model_name}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    saving_path = Path(args.saving_path)
    with (saving_path / f'evaluation_{run_name}.json').open('w', encoding='utf-8') as handle:
        json.dump({'Accuracy': acc, 'num_samples': len(samples), 'args': vars(args)}, handle, indent=2)
    with (saving_path / f'predictions_{run_name}.json').open('w', encoding='utf-8') as handle:
        json.dump([
            {'sample': sample, 'prediction': pred.outputs[0].text, 'context': pred.prompt}
            for sample, pred in zip(samples, preds)
        ], handle, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main(paras_args())
