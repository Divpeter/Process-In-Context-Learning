"""Prompt construction for PICL reasoning and confusion detection."""

import re
from typing import Dict, List

from dsicl.template import CONFUSION_DETECTION_PREFIX, CONFUSION_DETECTION_SUFFIX


def extract_confusion(text: str) -> str:
    """Read a nonempty confusion summary, ignoring a detector's private reasoning."""
    response = text.rsplit('</think>', 1)[-1]
    match = re.search(r'\bconfusion\s*\{', response, flags=re.IGNORECASE)
    if not match:
        return ''
    decision = re.search(r'\b(yes|no)\b', response[:match.start()], flags=re.IGNORECASE)
    if decision and decision.group(1).lower() == 'no':
        return ''
    start = match.end()
    depth = 1
    for index in range(start, len(response)):
        if response[index] == '{':
            depth += 1
        elif response[index] == '}':
            depth -= 1
            if depth == 0:
                return response[start:index].strip()
    return ''


class Prompter:
    def __init__(self, template: str, head: str, sep: str = '\n', label_map=None, task=None):
        self.template = template
        self.prompting_head = head
        self.sep = sep
        self.label_map = label_map
        self.task = task

    def _format_choices(self, choices):
        if not isinstance(choices, (list, tuple)):
            return str(choices)
        labels = range(len(choices)) if self.task == 'mmlu_math' else (
            chr(ord('A') + index) for index in range(len(choices))
        )
        return '\n'.join(f'({label}) {choice}' for label, choice in zip(labels, choices))

    def generate_label_prompt(self, demo: Dict):
        values = dict(demo)
        if self.label_map and 'label' in values:
            values['label'] = self.label_map[values['label']]
        if 'choices' in values:
            values['choices'] = self._format_choices(values['choices'])
        lines = []
        for line in self.template.splitlines():
            fields = re.findall(r'\[([A-Za-z_]+)\]', line)
            if all(field in values for field in fields):
                lines.append(re.sub(r'\[([A-Za-z_]+)\]', lambda m: str(values[m.group(1)]), line))
        return self.sep.join(lines)

    def generate_prompt(self, sample: Dict):
        """Format the question only; never expose a target answer or solution."""
        for key in ('question', 'problem', 'Problem'):
            if key in sample:
                prompt = f"Question: {sample[key]}"
                break
        else:
            raise ValueError('A sample must contain question, problem, or Problem.')
        if 'choices' in sample:
            prompt += f"\nChoices:\n{self._format_choices(sample['choices'])}"
        return prompt

    def generate_context(self, demonstrations: List[Dict], sample: Dict):
        context = self.prompting_head.rstrip() + '\n<|im_start|>user\n'
        if demonstrations:
            context += 'Examples:\n'
            context += self.sep.join(self.generate_label_prompt(demo) for demo in demonstrations)
            context += self.sep
        context += self.generate_prompt(sample)
        return context + '<|im_end|>\n<|im_start|>assistant\n'

    def generate_confusion_query(self, contexts):
        prompts = []
        for context in contexts:
            # Put the question and partial reasoning inside a single detector message.
            sequence = context.split('<|im_start|>user\n', 1)[-1]
            sequence = re.sub(r'<\|im_start\|>(?:system|user|assistant)\s*', '\n', sequence)
            sequence = sequence.replace('<|im_end|>', '').strip()
            prompts.append(
                f'<|im_start|>user\n{CONFUSION_DETECTION_PREFIX} '
                f'{sequence}. {CONFUSION_DETECTION_SUFFIX}'
                '<|im_end|>\n<|im_start|>assistant\n'
            )
        return prompts
