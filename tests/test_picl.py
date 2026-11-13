"""Offline regression tests for the paper prompts and the PICL execution path."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from PICL import Evaluate, Semantic_retrieval, run_Picl
from dsicl.data_reader import DataReader
from dsicl.inferencer.direct_inferencer import PICLInferencer
from dsicl.picl_retriever import PiclRetriever
from dsicl.prompter import Prompter, extract_confusion
from dsicl.retriever.embedding_retriever import EmbeddingRetriever
from dsicl.template import DEMO_TEMPLATE, PICL_HEAD


def prompter(task='gsm8k'):
    return Prompter(DEMO_TEMPLATE[task], PICL_HEAD[task], task=task)


def completion(text, prompt=''):
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)], prompt=prompt)


class FakeTokenizer:
    eos_token = '<eos>'

    def encode(self, text):
        return list(text)


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, contexts, sampling_params):
        assert contexts, 'An empty batch must not reach the model.'
        texts = self.responses.pop(0)
        assert len(contexts) == len(texts)
        self.calls.append((list(contexts), sampling_params))
        return [completion(text, context) for text, context in zip(texts, contexts)]


class PromptTests(unittest.TestCase):
    def test_start_prompt_matches_figure_10(self):
        actual = prompter().generate_context([], {
            'question': 'What is 1 + 1?', 'answer': 'SECRET_SOLUTION', 'label': 'SECRET_LABEL',
        })
        self.assertEqual(actual,
            '<|im_start|>system\n'
            'Please reason step by step, and put your final answer within \\boxed{}.'
            '<|im_end|>\n<|im_start|>user\nQuestion: What is 1 + 1?'
            '<|im_end|>\n<|im_start|>assistant\n')

    def test_all_task_queries_hide_targets_and_keep_choices(self):
        for task in DEMO_TEMPLATE:
            with self.subTest(task=task):
                sample = {
                    'question': 'Find x.', 'problem': 'Find x.', 'Problem': 'Find x.',
                    'answer': 'SECRET', 'Answer': 'SECRET', 'label': 'SECRET',
                    'solution': 'SECRET', 'Solution': 'SECRET',
                }
                query = prompter(task).generate_prompt(sample)
                self.assertEqual(query, 'Question: Find x.')
                self.assertNotIn('SECRET', query)
        sample = {'question': 'Pick one.', 'choices': ['red', 'blue'], 'answer': 1}
        self.assertIn('(0) red\n(1) blue', prompter('mmlu_math').generate_prompt(sample))
        sample['answer'] = 'B'
        self.assertIn('(A) red\n(B) blue', prompter('mmlu_pro').generate_prompt(sample))

    def test_demo_preserves_solution_and_real_linebreaks(self):
        demo = {'problem': 'Find x.', 'solution': 'x = 2.', 'answer': '2'}
        self.assertEqual(
            prompter('math500').generate_label_prompt(demo),
            'Problem: Find x.\nSolution: x = 2.\nAnswer: 2',
        )

    def test_detection_uses_process_wording_and_one_chat_message(self):
        context = prompter().generate_context([], {'question': 'Find x.'}) + 'I am unsure. Wait'
        prompt = prompter().generate_confusion_query([context])[0]
        self.assertIn('Answer with "Yes" if this process appears confused', prompt)
        self.assertIn('"No" if the process does not appear confused.', prompt)
        self.assertIn('Find x.', prompt)
        self.assertIn('I am unsure. Wait', prompt)
        self.assertIn('format confusion{ }.', prompt)
        self.assertNotIn('student', prompt)
        self.assertEqual(prompt.count('<|im_start|>user'), 1)
        self.assertEqual(prompt.count('<|im_start|>assistant'), 1)
        self.assertEqual(prompt.count('<|im_end|>'), 1)

    def test_confusion_requires_a_nonempty_summary(self):
        for text in ('Yes', 'yesterday', 'No', 'Yes confusion{}', 'Yes confusion{  }',
                     'No. confusion{example only}', 'Yes confusion{unclosed'):
            with self.subTest(text=text):
                self.assertEqual(extract_confusion(text), '')
        self.assertEqual(extract_confusion(
            '<think>No, maybe yes confusion{discard}</think>Yes. Confusion{\nUse x^{2}.\n}'
        ), 'Use x^{2}.')


class RetrievalTests(unittest.TestCase):
    def test_l2_default_and_cosine_rankings_exclude_query_before_top_n(self):
        data = DataReader([
            {'question': 'same', 'answer': 'different metadata'},
            {'question': 'parallel'},
            {'question': 'nearby'},
            {'question': 'same', 'answer': 'another duplicate'},
        ])
        encoder = lambda _: torch.tensor([1., 0.])
        with patch.dict(sys.modules, {'dsicl.model': SimpleNamespace(get_encoder=lambda *a, **kw: encoder)}):
            retriever = EmbeddingRetriever('local-encoder', prompter(), data)
        retriever.embeddings = torch.tensor([[1., 0.], [10., 0.], [1., 0.5], [1., 0.]])
        demos = retriever.retrieve({'question': 'same', 'answer': 'target'}, 2, exclude_query=True)
        self.assertEqual([demo['question'] for demo in demos], ['nearby', 'parallel'])
        retriever.similarity_metric = 'cosine'
        demos = retriever.retrieve({'question': 'same', 'answer': 'target'}, 2, exclude_query=True)
        self.assertEqual([demo['question'] for demo in demos], ['parallel', 'nearby'])
        retriever.similarity_metric = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'similarity_metric'):
            retriever.retrieve({'question': 'same'}, 2)

    def test_reranker_receives_full_demo_then_confusion_and_keeps_best_first(self):
        calls = []
        def compute_score(pairs, normalize):
            calls.append(pairs)
            return [0.2, 0.9]
        retriever = PiclRetriever.__new__(PiclRetriever)
        retriever.prompter = prompter('math500')
        retriever.reranker = SimpleNamespace(
            tokenizer=SimpleNamespace(sep_token='</s>'), compute_score=compute_score,
        )
        demos = [
            {'problem': 'First', 'solution': 'First solution', 'answer': '1'},
            {'problem': 'Second', 'solution': 'Second solution', 'answer': '2'},
        ]
        selected = retriever.confusion_reranking(
            {'problem': 'Query', 'answer': 'HIDDEN'}, completion('Yes. confusion{Need a formula}'), demos, 2,
        )
        self.assertEqual(selected.to_list(), demos[::-1])
        self.assertEqual(calls[0][0][0], 'Question: Query')
        self.assertIn('Solution: First solution\nAnswer: 1</s>Need a formula', calls[0][0][1])

    def test_reranker_handles_one_or_zero_candidates(self):
        retriever = PiclRetriever.__new__(PiclRetriever)
        retriever.prompter = prompter()
        retriever.reranker = SimpleNamespace(
            tokenizer=SimpleNamespace(sep_token='</s>'), compute_score=lambda *a, **kw: 0.7,
        )
        demos = [{'question': 'Demo', 'answer': 'Solution'}]
        detection = completion('Yes. confusion{Need help}')
        self.assertEqual(retriever.confusion_reranking(
            {'question': 'Query'}, detection, demos, 4).to_list(), demos)
        self.assertEqual(retriever.confusion_reranking(
            {'question': 'Query'}, detection, [], 4).to_list(), [])


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.vllm = patch.dict(sys.modules, {'vllm': SimpleNamespace(SamplingParams=SimpleNamespace)})
        self.vllm.start()
        self.addCleanup(self.vllm.stop)
        self.args = SimpleNamespace(
            max_model_len=10000, max_tokens=20000, r=1, k=2, m=None,
            reranker_model_path='local-reranker',
        )

    def inferencer(self, model):
        return PICLInferencer(model, FakeTokenizer(), prompter(), [], self.args)

    def test_finished_batch_never_runs_detector_or_reranker(self):
        model = ScriptedModel([[r'\boxed{2}']])
        with patch('dsicl.inferencer.direct_inferencer.PiclRetriever') as retriever:
            result = self.inferencer(model).batch_infer_picl([[]], [{'question': '1 + 1?'}])[0]
        self.assertEqual(result[0].outputs[0].text, r'\boxed{2}')
        self.assertEqual(len(model.calls), 1)
        retriever.assert_not_called()

    def test_mixed_batch_inserts_only_for_confused_question_and_preserves_order(self):
        model = ScriptedModel([
            [r'\boxed{2}', 'Let me check. Wait'],
            ['Yes. confusion{Which base quantity?}'],
            [r'\boxed{3}'],
        ])
        demo = {'question': 'Demo question', 'answer': 'Demo solution', 'label': '3'}
        with patch('dsicl.inferencer.direct_inferencer.PiclRetriever') as retriever:
            retriever.return_value.batch_confusion_reranking.return_value = [DataReader([demo])]
            results = self.inferencer(model).batch_infer_picl(
                [[], [demo]], [{'question': 'First question'}, {'question': 'Second question'}],
            )[0]
            rerank_args = retriever.return_value.batch_confusion_reranking.call_args.args
            self.assertEqual(rerank_args[2], [[demo]])
        self.assertEqual([result.outputs[0].text for result in results], [r'\boxed{2}', r'\boxed{3}'])
        self.assertNotIn('Demo solution', model.calls[0][0][1])
        self.assertIn('Demo solution', model.calls[-1][0][0])
        self.assertIn('Second question', model.calls[1][0][0])
        self.assertNotIn('First question', model.calls[1][0][0])

    def test_yes_without_summary_resumes_without_insertion(self):
        model = ScriptedModel([[' Wait'], ['Yes'], [r'\boxed{2}']])
        with patch('dsicl.inferencer.direct_inferencer.PiclRetriever') as retriever:
            self.inferencer(model).batch_infer_picl([[]], [{'question': '1 + 1?'}])
        retriever.assert_not_called()
        self.assertEqual(len(model.calls), 3)

    def test_intervention_budget_limits_detection_rounds(self):
        self.args.r = 2
        model = ScriptedModel([
            [' Wait'], ['No'], [' wait'], ['No'], [r'\boxed{2}'],
        ])
        self.inferencer(model).batch_infer_picl([[]], [{'question': '1 + 1?'}])
        self.assertEqual(len(model.calls), 5)
        self.assertNotIn(' Wait', model.calls[-1][1][0].stop)

    def test_context_budget_caps_generation_and_rejects_overflow(self):
        model = ScriptedModel([['done']])
        inferencer = self.inferencer(model)
        inferencer.max_model_len = 12
        inferencer._generate(['12345'])
        self.assertEqual(model.calls[0][1][0].max_tokens, 7)
        with self.assertRaisesRegex(ValueError, 'context exceeds'):
            inferencer._generate(['x' * 12])


class EntryPointTests(unittest.TestCase):
    def test_retrieval_cache_feeds_evaluation_with_a_sample_limit(self):
        samples = [
            {'problem': 'First question', 'solution': 'First solution', 'answer': '2'},
            {'problem': 'Second question', 'solution': 'Second solution', 'answer': '3'},
        ]
        encoder = lambda text: torch.tensor([1., 0.]) if 'First' in text else torch.tensor([0., 1.])
        model = ScriptedModel([[r'\boxed{2}']])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            (root / 'math500.json').write_text(json.dumps({
                'data': samples, 'data_info': {'label_space': []},
            }), encoding='utf-8')
            with patch.dict(sys.modules, {'dsicl.model': SimpleNamespace(get_encoder=lambda *a, **kw: encoder)}):
                Semantic_retrieval.main(Semantic_retrieval.paras_args([
                    '--data_path', str(root), '--saving_path', str(output),
                    '--task', 'math500', '--shots', '1',
                ]))
            pairs = Evaluate.load_sample_demos_pairs(output, 'math500')
            self.assertEqual(pairs[0]['demos'], [samples[1]])
            self.assertEqual(pairs[1]['demos'], [samples[0]])

            fake_modules = {
                'transformers': SimpleNamespace(
                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: FakeTokenizer()),
                ),
                'vllm': SimpleNamespace(LLM=lambda **kw: model, SamplingParams=SimpleNamespace),
            }
            with patch.dict(sys.modules, fake_modules), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.device_count', return_value=1), \
                    patch('dsicl.utils.set_seed'):
                Evaluate.main(Evaluate.paras_args([
                    '--saving_path', str(output), '--model_path', 'local-model',
                    '--task', 'math500', '--m', '1',
                ]))
            evaluation_files = list(output.glob('evaluation_*.json'))
            self.assertEqual(len(evaluation_files), 1)
            evaluation = json.loads(evaluation_files[0].read_text(encoding='utf-8'))
            self.assertEqual(evaluation['num_samples'], 1)
            self.assertEqual(evaluation['Accuracy'], 1.0)
            prediction_file = next(output.glob('predictions_*.json'))
            predictions = json.loads(prediction_file.read_text(encoding='utf-8'))
            self.assertEqual(len(predictions), 1)
            self.assertNotIn('First solution', predictions[0]['context'])

    def test_stage_commands_use_current_python_and_valid_stage_arguments(self):
        for stage in ('all', 'retrieval', 'picl'):
            with self.subTest(stage=stage):
                args = run_Picl.parse_args([
                    '--stage', stage, '--saving_path', 'output with spaces', '--task', 'math500',
                    '--data_path', 'data with spaces', '--model_path', 'model with spaces', '--m', '7',
                ])
                commands = run_Picl.build_commands(args)
                self.assertEqual(len(commands), 2 if stage == 'all' else 1)
                for command in commands:
                    self.assertEqual(command[0], sys.executable)
                    self.assertTrue(Path(command[1]).is_absolute())
                    self.assertTrue(Path(command[1]).is_file())
                    if Path(command[1]).name == 'Semantic_retrieval.py':
                        parsed = Semantic_retrieval.paras_args(command[2:])
                        self.assertEqual(parsed.shots, 32)
                    else:
                        parsed = Evaluate.paras_args(command[2:])
                        self.assertEqual(parsed.m, 7)
                    self.assertEqual(parsed.task, 'math500')

    def test_invalid_parameters_fail_before_starting_jobs(self):
        for extra in (['--k', '0'], ['--r', '0'], ['--m', '-1'], ['--k', '33']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run_Picl.parse_args([
                        '--saving_path', 'out', '--data_path', 'data', '--model_path', 'model', *extra,
                    ])

    def test_failed_retrieval_stops_pipeline(self):
        with patch.object(run_Picl.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'retrieval')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                run_Picl.main(['--saving_path', 'out', '--data_path', 'data', '--model_path', 'model'])
        self.assertEqual(run.call_count, 1)

    def test_cache_rejects_wrong_task_and_empty_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'retrieval_args.json').write_text(json.dumps({'task': 'gsm8k'}), encoding='utf-8')
            (root / 'demos.json').write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'not math500'):
                Evaluate.load_sample_demos_pairs(root, 'math500')
            with self.assertRaisesRegex(ValueError, 'no questions'):
                Evaluate.load_sample_demos_pairs(root, 'gsm8k')


if __name__ == '__main__':
    unittest.main()
