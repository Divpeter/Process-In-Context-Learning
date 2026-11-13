"""Build the semantic candidate cache used by PICL."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def paras_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--saving_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--task', default='gsm8k')
    parser.add_argument('--shots', type=int, default=32, help='Number of retrieval candidates N')
    parser.add_argument('--embedding_model_path', default='bert-base-uncased')
    parser.add_argument('--seed', type=int, default=317)
    args = parser.parse_args(argv)
    if args.shots < 1:
        parser.error('--shots must be positive')
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from tqdm import tqdm
    from dsicl.utils import set_seed
    from dsicl.data_reader import read_demo_benchmark
    from dsicl.prompter import Prompter
    from dsicl.template import DEMO_TEMPLATE, PICL_HEAD
    from dsicl.retriever.embedding_retriever import EmbeddingRetriever

    set_seed(args.seed)
    pool, testset = read_demo_benchmark(data_path=args.data_path, task=args.task)
    if not pool or not testset:
        raise ValueError('The demonstration pool and test set must not be empty.')
    prompter = Prompter(DEMO_TEMPLATE[args.task], PICL_HEAD[args.task], task=args.task)
    retriever = EmbeddingRetriever(args.embedding_model_path, prompter, pool)
    retriever.build_embeddings()
    pairs = [
        {'sample': sample, 'demos': retriever.retrieve(sample, args.shots, exclude_query=True).to_list()}
        for sample in tqdm(testset, desc='Retrieving candidates')
    ]

    saving_path = Path(args.saving_path)
    saving_path.mkdir(parents=True, exist_ok=True)
    with (saving_path / 'demos.json').open('w', encoding='utf-8') as handle:
        json.dump(pairs, handle, indent=2, ensure_ascii=False)
    with (saving_path / 'retrieval_args.json').open('w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, indent=2)
    print(f'Saved candidates for {len(pairs)} questions to {saving_path / "demos.json"}')


if __name__ == '__main__':
    main(paras_args())
