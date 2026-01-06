import json
from typing import List, Union, Optional, Dict, Tuple
import logging
import numpy as np
#from template import PROMPTING_HEAD, TEMPLATE, SENTENCE_HEAD
import functools

class Prompter:

    def __init__(self, template: str, head: str, sep: str = '\n', label_map = None) -> None:
        self.template = template
        self.prompting_head = head
        self.sep = sep
        self.label_map = label_map

    def generate_label_prompt(self, demo: Dict):
        if self.label_map is None:
            if 'label' in demo.keys():
                prompt = self.template.replace('[label]',demo['label'])
            elif 'answer' in demo.keys():
                prompt = self.template.replace('[answer]',str(demo['answer']))
            elif 'Answer' in demo.keys():
                prompt = self.template.replace('[Answer]',str(demo['Answer']))
        else:
            prompt = prompt = self.template.replace('[label]',self.label_map[demo['label']])
        for key, value in demo.items():
            prompt = prompt.replace(f'[{key}]', str(value))
        return prompt

    def generate_prompt(self, sample: Dict):
        prompt = self.template.replace('[label]','')
        for key, value in sample.items():
            if key != 'answer' and key != 'solution' and key != 'Answer' and key != 'Solution':
                prompt = prompt.replace(f'[{key}]', str(value))
            else:
                prompt = prompt.replace(f'[{key}]', '')
                prompt = prompt.replace('\nLabel:', '')
                prompt = prompt.replace('\nAnswer:', '')
        return prompt

    def generate_context(self, demonstrations: List[Dict], sample: Dict):

        context = self.prompting_head+self.sep
        context += "<|im_start|>user"
        if len(demonstrations) != 0:
            context += "Examples: "
        for demo in demonstrations:
            context += self.generate_label_prompt(demo)
            context += self.sep
        context += "Question: "
        context += self.generate_prompt(sample)
        context += "<|im_end|>\n"
        context += "<|im_start|>assistant\n"
        return context
    
    def generate_confusion_query(self, context):
        # prefix = """You are a math problem-solving expert. Here is a student's intermediate problem-solving process for a math question. 
        # Please analyze the student's current solution process and determine whether the student has shown any signs of confusion at this stage.""" 
        # lastfix = """Remember your job is to classify whether the student is confused or not as quickly as possible and you don't need to answer the quesiton yourself. 
        # The process should be quick and accurate. Remember that your answer should be decisive and quick. Answer with "Yes" if the student appears confused, and "No" if the student does not appear confused. 
        # If your answer is "Yes", you need to summarize the current confusion and write it in the format confusion{ }. If there is no confusion, simply answer "No". Remember the confusion should be written in the format confusion{ }"""

        prefix2 = """Please analyze the intermediate problem-solving process for a math question and determine whether it has shown any signs of confusion at this
                stage. Answer with "Yes" if the student appears confused, and "No" if the student does not appear confused.""" 
        lastfix2 = """If your answer is "Yes", you need to summarize the current confusion and write it in the format confusion{ }."""
        new_context = [f"<|im_start|>user{prefix2}: {item}. {lastfix2}<|im_end|> <|im_start|>assistant" for item in context]
        
        return new_context
