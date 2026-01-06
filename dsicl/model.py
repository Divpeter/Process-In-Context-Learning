import torch
import logging
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

MODEL_ROOT = 'XXX'

class encoder(nn.Module):
    def __init__(self, model, device):
        super(encoder, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.encoder = AutoModel.from_pretrained(model).to(device)

    def forward(self, x):
        with torch.no_grad(): 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(x, return_tensors='pt' ,max_length=512,  truncation=True).to(self.device)
            output = self.encoder(**inputs)
            return output['pooler_output'][0]

def get_encoder(model, device):
    return encoder(model, device)

class speculative_decoder:
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, truncation_l = 20480):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = self.model.device
        self.log_soft = nn.LogSoftmax(dim=0)
        self.soft = nn.Softmax(dim=0)
        self.truncation_l = truncation_l
        
    def decode(self, prefix:str, answer_list:list, return_logits=False):

        with torch.no_grad():

            prompts = [prefix+answer for answer in answer_list] # 拼接所有候选label
            
            l = len(self.tokenizer(prefix,max_length=self.truncation_l,truncation=True, return_tensors="pt").input_ids[0]) # prefix长度
            
            inputs = self.tokenizer(prompts,max_length=self.truncation_l, truncation=True, padding= True, return_tensors="pt").to(self.model.device)
            
            seq_logits = self.model(**inputs, use_cache=False).logits[:, l-1:-1,:]
            
            tokens = inputs.input_ids[:,l:] # 取label对应的logits

            log_probs = torch.zeros(len(prompts), device=self.model.device)

            for i in range(len(prompts)):
                for j in range(seq_logits.size()[1]):
                    token_id = tokens[i][j]
                    if token_id == self.tokenizer.eos_token_id: # 遇到padding token结束
                        break
                    vocabulary = seq_logits[i][j]
 
                    log_probs[i] += self.log_soft(vocabulary)[token_id]
            probs = self.soft(log_probs)
            torch.cuda.empty_cache()
            if return_logits:
                return log_probs
            return probs
    