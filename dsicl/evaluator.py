import torch
import numpy as np
import json
from typing import List, Tuple, Dict
from tqdm import trange, tqdm
from collections import defaultdict
from dsicl.math_equivalence import *
import re

class Evaluator:
    def __init__(self) -> None:
        pass

    def evaluate_predictions(self, preds, labeled_answer, task,  mode='gen'):
        output = []
        for idx in range(len(preds)):
            output.append(preds[idx].outputs[0].text)
        # output = preds
        final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
        pred_answer = extract_answer(output, mode=mode)
        if pred_answer != '':
            final_metric["is_valid_answer"] = True

        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)

        s_equal = 0
        #applied for complex answers
        m_equal = 0
        for pred, label in zip(normalized_pred_answer, normalized_ground_truth):
            if label in pred:
                s_equal += 1
            if is_equiv(pred, label):
                m_equal += 1
        final_metric["s_equal"] = s_equal/(len(normalized_pred_answer))
        final_metric["m_equal"] = m_equal/(len(normalized_pred_answer))

        if task in ['gsm8k']:
            acc = final_metric["s_equal"]
        else:
            acc = final_metric["m_equal"]
    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
        return acc, pred_answer, normalized_pred_answer, normalized_ground_truth

    def evaluate_selections(self, preds, labeled_answer, mode='gen'):
        output = []
        for idx in range(len(preds)):
            output.append(preds[idx].outputs[0].text)
        # output = preds
        final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
        pred_answer = extract_answer(output, mode=mode)
        if pred_answer != '':
            final_metric["is_valid_answer"] = True

        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth_o = normalize_answer(labeled_answer[0])
        normalized_ground_truth_c = normalize_answer(labeled_answer[1])
        normalized_ground_truth_a = normalize_answer(labeled_answer[2])

        acc = 0
        for pred, labelo, labelc, labela in zip(normalized_pred_answer, normalized_ground_truth_o, normalized_ground_truth_c, normalized_ground_truth_a):
            if labelo in pred:
                acc += 1
            elif labelc in pred:
                acc +=1
            elif labela in pred:
                acc +=1
        final_metric["acc"] = acc/(len(normalized_pred_answer))
        acc = final_metric["acc"]
    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
        return acc


    def acc_evaluate(self, y_pred: List[str], y_true: List[str]) -> float:
        acc = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_pred)
        # if confusion_matrix:

        return float(acc)

    def f1_evaluate(self, y_pred: List[str], y_true: List[str]) -> float:
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)

        for yp, yt in zip(y_pred, y_true):
            if yp == yt:
                true_positives[yp] += 1
            else:
                false_positives[yp] += 1
                false_negatives[yt] += 1

        f1_scores = []
        for label in set(y_pred).union(set(y_true)):
            tp = true_positives[label]
            fp = false_positives[label]
            fn = false_negatives[label]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)

        if f1_scores:
            return sum(f1_scores) / len(f1_scores)
        else:
            return 0.0

    def std(self, values: List[float]) -> float:
        if not values:
            return 0.0

        mean = sum(values) / len(values)

        variance = sum((x - mean) ** 2 for x in values) / len(values)

        return float((np.array(variance)**0.5))

    def mean(self, values: List[float]) -> float:
        if not values:
            return 0.0

        mean = sum(values) / len(values)

        return mean

# def extract_number(text):
#     numbers = re.findall(r'-?\d+\.\d+|-?\d+|frac\{(-?\d+)\}/\{(-?\d+)\}', text)

#     processed_numbers = []

#     for num in numbers:
#             numerator = int(num[0])
#             denominator = int(num[1])
#             processed_numbers.append(numerator / denominator)
#             processed_numbers.append(float(num))

#     last_number = processed_numbers[-1] if processed_numbers else None
#     return last_number

def extract_answer(output, mode='gen'):
    pattern = r'\\boxed\{(.*)\}'
    # number_pattern = r'-?\d+(?:\.\d+)?'
    number_pattern = r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?'

    extracted_text_output = []
    undetected = 0
    for text in output:
        matches = re.findall(pattern, text)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            # if mode in ['choose', 'qa']:
            #     # Handle 'choose' mode
            #     inner_pattern = r'\\text\{(.*)\}'
            #     inner_matches = re.findall(inner_pattern, extracted_text)
            #     if inner_matches:
            #         extracted_text = inner_matches[-1]  # Take the last match
            #     extracted_text = extracted_text.strip("()")
            extracted_text_output.append(extracted_text)
        else:
            number_matches = re.findall(number_pattern, text)
            if number_matches:
                extracted_text = number_matches[-1]
                extracted_text_output.append(extracted_text)
            else:
                undetected += 1
                extracted_text_output.append('0')
    print(f"{undetected} answers don't match")
    return extracted_text_output

def is_number(val):
    return isinstance(val, (int, float))

def normalize_answer(text):
    normalized_text = []

    for ele in text:
        if isinstance(ele, list):
            ele = ele[0]
        if not is_number(ele):
            ele = ele.lower()
            ele = " ".join(ele.strip().split())
        else:
            ele = str(ele)
        normalized_text.append(ele)
    return normalized_text
