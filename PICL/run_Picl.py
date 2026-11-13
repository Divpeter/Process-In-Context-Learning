"""Run semantic retrieval and PICL inference, together or separately."""

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'retrieval', 'picl'), default='all')
    parser.add_argument('--gpu', default='0', help='Comma-separated GPU IDs')
    parser.add_argument('--task', default='gsm8k', help='Benchmark name')
    parser.add_argument('--data_path', help='Directory containing benchmark JSON files')
    parser.add_argument('--saving_path', required=True, help='Candidate and result directory')
    parser.add_argument('--embedding_model_path', default='bert-base-uncased')
    parser.add_argument('--retrieval_shots', type=int, default=32, help='Candidate count N')
    parser.add_argument('--model_path', help='Reasoning model directory or model ID')
    parser.add_argument('--model_name', help='Optional name used in result filenames')
    parser.add_argument('--reranker_model_path', default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--r', type=int, default=1, help='Maximum interventions per question')
    parser.add_argument('--k', type=int, default=1, help='Demonstrations per intervention')
    parser.add_argument('--m', type=int, help='Evaluate only the first m questions; default: all')
    parser.add_argument('--max_model_len', type=int, default=20000)
    parser.add_argument('--max_tokens', type=int, default=20000, help='Maximum new tokens per call')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.83)
    parser.add_argument('--seed', type=int, default=317)
    args = parser.parse_args(argv)
    if args.stage in ('all', 'retrieval') and not args.data_path:
        parser.error('--data_path is required for semantic retrieval')
    if args.stage in ('all', 'picl') and not args.model_path:
        parser.error('--model_path is required for PICL inference')
    if args.retrieval_shots < 1 or args.k < 1 or args.r < 1:
        parser.error('--retrieval_shots, --k, and --r must be positive')
    if args.stage == 'all' and args.k > args.retrieval_shots:
        parser.error('--k cannot exceed --retrieval_shots')
    if args.m is not None and args.m < 1:
        parser.error('--m must be positive')
    if args.max_model_len < 2 or args.max_tokens < 1:
        parser.error('--max_model_len must be at least 2 and --max_tokens must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('--gpu_memory_utilization must be between 0 and 1')
    return args


def build_commands(args):
    script_dir = Path(__file__).resolve().parent
    common = ['--gpu', args.gpu, '--task', args.task, '--saving_path', args.saving_path]
    commands = []
    if args.stage in ('all', 'retrieval'):
        commands.append([
            sys.executable, str(script_dir / 'Semantic_retrieval.py'), *common,
            '--data_path', args.data_path,
            '--embedding_model_path', args.embedding_model_path,
            '--shots', str(args.retrieval_shots), '--seed', str(args.seed),
        ])
    if args.stage in ('all', 'picl'):
        command = [
            sys.executable, str(script_dir / 'Evaluate.py'), *common,
            '--model_path', args.model_path,
            '--reranker_model_path', args.reranker_model_path,
            '--r', str(args.r), '--k', str(args.k),
            '--max_model_len', str(args.max_model_len),
            '--max_tokens', str(args.max_tokens),
            '--gpu_memory_utilization', str(args.gpu_memory_utilization),
            '--seed', str(args.seed),
        ]
        if args.model_name:
            command.extend(['--model_name', args.model_name])
        if args.m is not None:
            command.extend(['--m', str(args.m)])
        commands.append(command)
    return commands


def main(argv=None):
    for command in build_commands(parse_args(argv)):
        print(f'Running {Path(command[1]).name}', flush=True)
        subprocess.run(command, check=True)
    print('PICL run completed successfully.')


if __name__ == '__main__':
    main()
