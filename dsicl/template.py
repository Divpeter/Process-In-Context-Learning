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
    'gsm8k':"Question:[question]\nAnswer:[answer]\nLabel:[label]",
    'aime':"ID:[ID]\nProblem:[Problem]\nSolution:[Solution]\nAnswer:[Answer]",
    'math500':"Problem:[problem]\Solution:[solution]\Answer:[answer]",
    'gsm8k_simplified':"Question:[question]\nAnswer:[answer]\nLabel:[label]",
    'sat_math':"Question:[question]\Choice:[choices]\Answer:[answer]",
    'olympiad':"Question:[question]\Solution:[solution]\Answer:[answer]",
    'amc':"Question:[question]\Solution:[solution]\Answer:[answer]",
    'mmlu_math':"Question:[question]\Choice:[choices]\Answer:[answer]",
    'mmlu_pro':"Question:[question]\Choice:[choices]\Answer:[answer]"
}

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

TEMPLATE = {
    'gsm8k':"Question:[question]\nAnswer:[answer]\nLabel:[label]",
    'aime':"ID:[ID]\nProblem:[Problem]\nSolution:[Solution]\nAnswer:[Answer]",
    'math500':"Problem:[problem]\Solution:[solution]\Answer:[answer]",
    'gsm8k_simplified':"Question:[question]\nAnswer:[answer]\nLabel:[label]",
    'sat_math':"Question:[question]\Choice:[choices]\Answer:[answer]",
    'olympiad':"Question:[question]\Solution:[solution]\Answer:[answer]",
    'amc':"Question:[question]\Solution:[solution]\Answer:[answer]",
    'mmlu_math':"Question:[question]\Choice:[choices]\Answer:[answer]",
    'mmlu_pro':"Question:[question]\Choice:[choices]\Answer:[answer]"
}

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
