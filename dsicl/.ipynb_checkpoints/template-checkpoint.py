MAX_SEARCH_LIMIT = 4

LABEL_MAP = {
    'agnews':{'Business':'Business', 'Technology':'Technology', 'World':'Technology', 'Sports':'Technology'},
    'boolq':{True:'True',False:'False'},
    'cb':{'entailment':'yes', 'neutral':'maybe', 'contradiction':'no'},
    'dbpedia':{'Album':'Album',
                'Animal':'Animal',
                'Artist':'Artist',
                'Athlete':'Athlete',
                'Book':'Book',
                'Building':'Building',
                'Company':'Company',
                'Film':'Film',
                'Nature':'Nature',
                'Plant':'Plant',
                'Politician':'Politician',
                'School':'School',
                'Transportation':'Transportation',
                'Village':'Village'},
    'imdb':{0: 'negative', 1: 'positive'},
    'mnli':{'entailment':'yes', 'neutral':'maybe', 'contradiction':'no'},
    'rte':{'entailment':'yes', 'not_entailment':'no'},
    'scicite':{0:'method', 1:'result', 2:'background'},
    'sst5':{0:'terrible', 1:'bad', 2:'okay', 3:'good', 4:'great'},
    'subj':{'objective':'objective', 'subjective':'subjective'},
    'trec':{'Abbreviation':'Abbreviation',
            'Description': 'Description',
            'Entity':'Entity',
            'Location': 'Location',
            'Number':'Number',
            'Person':'Person'},
    'yahoo':{ 0:'Society',
              1:'Science', 
              2:'Health',
              3:'Education', 
              4:'Computers',
              5:'Sports',
              6:'Business', 
              7:'Music',
              8:'Family',
              9:'Politics'},
    'gsm8k':{},
    'aime':{},
    'math500':{},
}

DEMO_TEMPLATE = {
    'sst2':"Review:[sentence]\nSentiment:[label]",
    'cr':"Review:[sentence]\nSentiment:[label]",
    'mr':"Review:[sentence]\nSentiment:[label]",
    'agnews':"Article:[sentence]\nAnswer:[label]",
    'trec':"Question:[sentence]\nAnswer:[label]",
    'subj':"Input:[sentence]\nLabel:[label]",  
    'rte':"Premise:[premise]\nHypothesis:[hypothesis]\nAnswer:[label]",  
    'snli':"Premise:[premise]\nHypothesis:[hypothesis]\nAnswer:[label]",
    'dbpedia':"Article:[sentence]\nAnswer:[label]", 
    'gsm8k':"Question:[question]\nAnswer:[answer]\nLabel:[label]",
    'aime':"ID:[ID]\nProblem:[problem]\nSolution:[solution]\nAnswer:[answer]",
    'math500':"Problem:[problem]\Solution:[solution]\Answer:[answer]"
}

math500_few_shot = "Please reason step by step, and put your final answer within \\boxed{} given provided examples. Focus on the underlying logic of the examples rather than getting too caught up in the specific content. Remember to put your answer in \\boxed{}"
math500_picl = "Please reason step by step, and put your final answer within \\boxed{}."
DEMO_HEAD = {
    'sst2':"Classify the reviews based on whether their sentiment type is positive or negative.",
    'agnews':"Classify the news based on whether their type is Sports, Business, Technology or World.",
    'trec':'Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.',
    'subj':"Classify the inputs based on whether their relationship type is yes, maybe or no.",    
    'mr':"Classify the reviews based on whether their sentiment type is positive or negative.", 
    'cr':"Classify the reviews based on whether their sentiment type is positive or negative.",   
    'rte':"Classify the pairs of premise and hypothesis based on whether the relationship type is yes or no.",   
    'snli':"Classify the pairs of premise and hypothesis based on whether the relationship type is yes, maybe or no.", 
    'dbpedia':"Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.", 
    'gsm8k':math500_few_shot,
    'aime':"Analyse the math question and give the final answer given the provided examples. Focus on the underlying logic of the examples rather than getting too caught up in the specific content.",
    'math500':math500_few_shot,
}

math500_0_shot = "Please reason step by step, and put your final answer within \\boxed{}."
math500_confusion = """Please reason step by step, and put your final answer within \\boxed{}. You are a reasoning assistant with the ability to perform In-context learning to help you solve your confusion point in the reasoning process. You have special tools:
 To perform an in-context learning: write <|begin_in_context_learning|> your confusion point here <|end_in_context_learning|>.
 Then, the system will search and analyze relevant in-context learning demonstrasions, then provide you with helpful examples in the
 format <|begin_icl_example|> ...searched example... <|end_icl_example|>.
 You can repeat the search process multiple times if necessary. The maximum number of search attempts is
 limited to 4.
 Once you have all the information you need, continue your reasoning.
 Example: "Ben's iPhone is two times older than Suzy's iPhone"
 Question: "how old is Brandon's iPhone?"
 Assistant confusion point:- "does that mean 2x or x + 2x = 3x? ..."
 Assistant:
 <|begin_in_context_learning|>...<|end_in_context_learning|>
 (System returns selected examples from example-pool)
 Assistant continues reasoning with the new information...
 Remember:- Use <|begin_in_context_learning|> to request in-context learning and end with <|end_in_context_learning|>.- When done in-context learning, continue your reasoning."""
math500_confusion_simple = """Please reason step by step, and put your final answer within \\boxed{}. You are a reasoning assistant with the ability to perform In-context learning to help you solve your confusion point in the reasoning process. You have special tools:
 To perform an in-context learning: write <|begin_in_context_learning|> your confusion point here <|end_in_context_learning|>.
 Then, the system will search and analyze relevant in-context learning demonstrasions, then provide you with helpful examples in the
 format <|begin_icl_example|> ...searched example... <|end_icl_example|>.
 You can repeat the search process multiple times if necessary. The maximum number of search attempts is
 limited to 4.
 Once you have all the information you need, continue your reasoning.
 Assistant:
 <|begin_in_context_learning|>...<|end_in_context_learning|>
 (System returns selected examples from example-pool)
 Assistant continues reasoning with the new information...
 Remember:- Use <|begin_in_context_learning|> to request in-context learning and end with <|end_in_context_learning|>.- When done in-context learning, continue your reasoning."""

HEAD = {
    'sst2':"Classify the reviews based on whether their sentiment type is positive or negative.",
    'rte':"Classify the pairs of premise and hypothesis based on whether the relationship type is yes or no.",
    'mnli':"Classify the pairs of premise and hypothesis based on whether the relationship type is yes, no or maybe.",
    'qnli':'Classify whether the sentence contains the answer to the question. The answer is yes or no.',
    'mrpc':"Classify whether the two sentences are semantically equivalent. The answer is yes or no.",     
    'cola':"Determine if the sentence is grammatically correct. The answer is yes or no.", 
    'gsm8k':math500_0_shot,
    'aime':"Answer the math question and give the final answer in a number",
    'math500': math500_0_shot
}

TEMPLATE = {
    'sst2':"Review:[sentence]\nSentiment:\n[label]",
    'rte':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:\n[label]",
    'mnli':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:\n[label]",
    'qnli':"Sentence:[sentence]\nQuestion:[question]\nLabel:\n[label]",
    'mrpc':"Sentence1:[sentence1]\nSentence2:[sentence2]\nLabel:\n[label]",
    'cola':"Sentence:[sentence]\nLabel:\n[label]",  
    'gsm8k':"Question:[question]\nAnswer:[answer]\nLabel:[label]",
    'aime':"ID:[ID]\nProblem:[problem]\nSolution:[solution]\nAnswer:[answer]",
    'math500':"Problem:[problem]\Solution:[solution]\Answer:[answer]"
}

LABEL = {
    'sst2':['negative','positive'],
    'rte':['yes','no'],
    'mnli':['yes','no','maybe'],
    'qnli':['yes','no'],
    'mrpc':['yes','no'],
    'cola':['yes','no'],
    'gsm8k':[],
    'aime':[],
    'math500':[],
}

# SENTENCE_HEAD = {
#     'sst2':"Review:",
#     'rte':"Premise:[premise]\nHypothesis:[hypothesis]",
#     'mnli':"Premise:[premise]\nHypothesis:[hypothesis]",
#     'qnli':"Sentence:[sentence]\nQuestion:[question]",
#     'mrpc':"Sentence1:[sentence1]\nSentence2:[sentence2]",
#     'cola':"Sentence:[sentence]",  
# }

# LABEL_HEAD = {
#     'sst2':"Review:[sentence]\nSentiment:\n[label]",
#     'rte':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:\n[label]",
#     'mnli':"Premise:[premise]\nHypothesis:[hypothesis]\nLabel:\n[label]",
#     'qnli':"Sentence:[sentence]\nQuestion:[question]\nLabel:\n[label]",
#     'mrpc':"Sentence1:[sentence1]\nSentence2:[sentence2]\nLabel:\n[label]",
#     'cola':"Sentence:[sentence]\nLabel:\n[label]",  
# }