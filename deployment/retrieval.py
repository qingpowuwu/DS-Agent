import os
import re
import json
import torch
import numpy as np
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel

DEVELOPMENT_TASKS = ["feedback", "airline-reviews", "textual-entailment", "chatgpt-prompt", "ett-m2", "ili", "handwriting", "ethanol-concentration", "media-campaign-cost", "wild-blueberry-yield", "spaceship-titanic", "enzyme-substrate"]
DEPLOYMENT_TASKS = ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat', 'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews', 'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1', 'uwave-gesture-library', 'traffic', 'boolq', 'crab-age', 'concrete-strength', 'jigsaw']

class RetrievalDatabase:
    def __init__(self, model="BAAI/llm-embedder") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        
        self.tokenizer = AutoTokenizer.from_pretrained(model)                          # 用 model 加载 tokenizer
        self.model = AutoModel.from_pretrained(model, 
                                              trust_remote_code=True).to(self.device) # 用 model 加载 model
        
        # Define query
        if model == "BAAI/llm-embedder":
            self.query_prompt = "Represent this query for retrieving relevant documents: "
            self.doc_prompt = "Represent this document for retrieval: "
        else:
            self.query_prompt = ""
            self.doc_prompt = ""

        # (1) 创建1个 case_bank 列表
        # Read cases from development tasks 并且 用 query_prompt 加上每个 case 的内容 => append 到 case_bank
        self.case_bank = []
        for task in DEVELOPMENT_TASKS:
            filename = f"../development/MLAgentBench/benchmarks/{task}/scripts/research_problem.txt"
            with open(filename) as file:
                self.case_bank.append(self.query_prompt + file.read()) # 把每个 case 的内容加上 query_prompt 后加入 case_bank
        # print('len(self.case_bank) = ', len(self.case_bank)) # 12
        
        # (2) 把 x_inputs 做 embedding: i.e, 把 self.case_bank 中的每个 case 用 tokenizer 处理后，得到 x_inputs
        # Construct Embedding Database
        x_inputs = self.tokenizer(
            self.case_bank,
            padding=True, 
            truncation= True,
            return_tensors='pt'
        )
            
        input_ids = x_inputs.input_ids.to(self.device)           # torch.Size([12, 190])
        attention_mask = x_inputs.attention_mask.to(self.device) # torch.Size([12, 190])

        # print('type(x_inputs) = ', type(x_inputs))   # <class 'transformers.tokenization_utils_base.BatchEncoding'>
        # print('x_inputs.keys() = ', x_inputs.keys()) #  dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        # print('x_inputs[''input_ids''].shape      = ', x_inputs['input_ids'].shape)      # torch.Size([12, 190])
        # print('x_inputs[''token_type_ids''].shape = ', x_inputs['token_type_ids'].shape) # torch.Size([12, 190])
        # print('x_inputs[''attention_mask''].shape = ', x_inputs['attention_mask'].shape) # torch.Size([12, 190])

        # (3) 用 self.model 预测 x_inputs 的输出 x_outputs : i.e, 把 input_ids 和 attention_mask 输入 self.model 得到 x_outputs
        with torch.no_grad():
            x_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            x_outputs = x_outputs.last_hidden_state[:, 0]
            x_embedding = torch.nn.functional.normalize(x_outputs, p=2, dim=1)
            
        self.embedding_bank = x_embedding

        # print('x_outputs.shape   = ', x_outputs.shape)    # torch.Size([12, 768])
        # print('x_embedding.shape = ', x_embedding.shape)  # torch.Size([12, 768])
        # print('self.embedding_bank.shape = ', self.embedding_bank.shape) # torch.Size([12, 768])
         
    
    def retrieve_case(self, query, num=12):
        # query: 用于检索的查询字符串, ex: "You are solving this data science tasks of binary classification: The dataset presented here (the Smoker Status Dataset) comprises a lot of numerical features. We have splitted the dataset into three parts of train, valid and test. Your task is to predict the smoking item, which is a binary label with 0 and 1. The evaluation metric is the area under ROC curve (AUROC). We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a binary classification model to get a good performance on this task."
        # num: 返回的最相似的 case 数量, 默认为 12

        x_inputs = self.tokenizer(
            self.query_prompt + query,
            padding=True, 
            truncation= True,
            return_tensors='pt'
        )
        input_ids = x_inputs.input_ids.to(self.device)
        attention_mask = x_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            x_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            x_outputs = x_outputs.last_hidden_state[:, 0]
            x_embedding = torch.nn.functional.normalize(x_outputs, p=2, dim=1)
        
        similarity = (x_embedding @ self.embedding_bank.T).squeeze() # torch.Size([12]), 
        #  ex: tensor([0.8704, 0.8977, 0.9105, 0.9118, 0.8381, 0.8380, 0.8849, 0.8751, 0.8763,
            
        _, ranking_index = torch.topk(similarity, num) # print('ranking_index.shape = ', ranking_index.shape) # torch.Size([12]),  tensor([ 4,  5,  9,  0,  7,  6,  8, 11,  1, 10,  3,  2], device='cuda:0')

        ranking_index = ranking_index.cpu().numpy().tolist()
        
        return [DEVELOPMENT_TASKS[i] for i in ranking_index] # ['chatgpt-prompt', 'textual-entailment', 'spaceship-titanic', 'airline-reviews', 'handwriting', 'ethanol-concentration', 'media-campaign-cost', 'wild-blueberry-yield', 'feedback', 'enzyme-substrate', 'ett-m2', 'ili']
    

if __name__ == '__main__':
    rb = RetrievalDatabase()
    ranking_dict = {}
    # 遍历 DEPLOYMENT_TASKS 中的 task, 用 retrieve_case 方法检索每个 task 的相似 case
    for task in DEPLOYMENT_TASKS: # smoker-status
        filename = f"../development/MLAgentBench/benchmarks/{task}/scripts/research_problem.txt"
        with open(filename) as file:
            query = file.read()  # 读取文件内容: query, You are solving this data science tasks of binary classification: 
                                            # The dataset presented here (the Smoker Status Dataset) comprises ...
            ranking_dict[task] = rb.retrieve_case(query) # 用 query 与 case_bank 中的每个 case 计算相似度，返回最相似的 12 个 case 的 task 名称 (用 rb 对象的 retrieve_case 方法)
            # ['enzyme-substrate', 'spaceship-titanic', 'airline-reviews', 'ethanol-concentration', 'chatgpt-prompt', 'handwriting', 'wild-blueberry-yield', 'media-campaign-cost', 'textual-entailment', 'feedback', 'ett-m2', 'ili']

    with open("./config/similarity_ranking.json", "wt") as json_file:
        print("Writing the ranking results to ./config/similarity_ranking.json")
        json.dump(ranking_dict, json_file, indent=4)
        
