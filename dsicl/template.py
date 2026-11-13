LABEL_MAP = {
    'gsm8k':{},
    'aime':{},
    'math500':{},
    'gsm8k_simplified':{},
    'sat_math':{},
    'olympiad':{},
    'amc':{},
    'mmlu_math':{},
    'mmlu_pro':{}
}

DEMO_TEMPLATE = {
    'gsm8k':"Question: [question]\nAnswer: [answer]\nLabel: [label]",
    'aime':"Problem: [Problem]\nSolution: [Solution]\nAnswer: [Answer]",
    'math500':"Problem: [problem]\nSolution: [solution]\nAnswer: [answer]",
    'gsm8k_simplified':"Question: [question]\nAnswer: [answer]\nLabel: [label]",
    'sat_math':"Question: [question]\nChoices: [choices]\nAnswer: [answer]",
    'olympiad':"Question: [question]\nSolution: [solution]\nAnswer: [answer]",
    'amc':"Question: [question]\nSolution: [solution]\nAnswer: [answer]",
    'mmlu_math':"Question: [question]\nChoices: [choices]\nAnswer: [answer]",
    'mmlu_pro':"Question: [question]\nChoices: [choices]\nAnswer: [answer]"
}

# Camera-ready Figure 10: PICL starts without static demonstrations.
PICL_START_PROMPT = (
    "<|im_start|>system\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
    "<|im_end|>\n"
)
PICL_HEAD = {task: PICL_START_PROMPT for task in DEMO_TEMPLATE}
CONFUSION_DETECTION_PREFIX = (
    "Please analyze the intermediate problem-solving process for a math question "
    "and determine whether it has shown any signs of confusion at this stage. "
    'Answer with "Yes" if this process appears confused, and "No" if the process '
    "does not appear confused."
)
CONFUSION_DETECTION_SUFFIX = (
    'If your answer is "Yes", you need to summarize the current confusion '
    "and write it in the format confusion{ }."
)

few_shot = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{} given provided examples.<|im_end|>\n "
math500_picl = "Please reason step by step, and put your final answer within \\boxed{}."
sat_math_few_shot = "<|im_start|>system\nPlease reason step by step, select the correct answer from (A) (B) (C) (D) given provided examples and put your final answer within \\boxed{}.<|im_end|>\n "
mmlu_math_few_shot = "<|im_start|>system\nPlease reason step by step, select the correct answer index from (0) (1) (2) (3) given provided examples and put your final answer within \\boxed{}.<|im_end|>\n "
mmlu_pro_few_shot = "<|im_start|>system\nPlease reason step by step, select the correct answer index given provided examples and put your final answer within \\boxed{}.<|im_end|>\n "
DEMO_HEAD = {
    'gsm8k':few_shot,
    'aime':few_shot,
    'math500':few_shot,
    'gsm8k_simplified':few_shot,
    'sat_math':sat_math_few_shot,
    'olympiad':few_shot,
    'amc':few_shot,
    'mmlu_math':mmlu_math_few_shot,
    'mmlu_pro':mmlu_pro_few_shot
}

zero_shot = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
sat_math_0_shot = "<|im_start|>system\nPlease reason step by step, select the correct answer from (A) (B) (C) (D) and put your final answer within \\boxed{}.<|im_end|>\n"
mmlu_math_0_shot = "<|im_start|>system\nPlease reason step by step, select the correct answer index from (0) (1) (2) (3) and put your final answer within \\boxed{}.<|im_end|>\n"
mmlu_pro_0_shot = "<|im_start|>system\nPlease reason step by step, select the correct answer index and put your final answer within \\boxed{}. <|im_end|>\n"
HEAD = {
    'gsm8k':zero_shot,
    'aime':zero_shot,
    'math500': zero_shot,
    'gsm8k_simplified':zero_shot,
    'sat_math':sat_math_0_shot,
    'olympiad':zero_shot,
    'amc':zero_shot,
    'mmlu_math':mmlu_math_0_shot,
    'mmlu_pro':mmlu_pro_0_shot
}

TEMPLATE = DEMO_TEMPLATE.copy()

LABEL = {
    'gsm8k':[],
    'aime':[],
    'math500':[],
    'gsm8k_simplified':[],
    'sat_math':[],
    'olympiad':[],
    'amc':[],
    'mmlu_math':[],
    'mmlu_pro':[]
}
