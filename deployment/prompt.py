import json
import random

RP_PATH = "./benchmarks/{}/scripts/research_problem.txt" # 要加载的 research problem 的路径
PYTHON_PATH = "./benchmarks/{}/env/train.py"
CASE_PATH = "./experience_replay/{}.py"

ZERO_SHOT_PROMPT = """
You are a helpful intelligent assistant. Now please help solve the following machine learning task.
[Task]
{}
[train.py] ```python
{}
```
Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
"""

FEW_SHOT_PROMPT = """
Here are some example cases that solve machine learning tasks:
{} 
Now please solve the following machine learning task based on the example cases above.
[Task]
{}
[train.py] ```python
{}
```
Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
"""

CASE_PROMPT = """[Task]
{}
[train.py] ```python
{}
```
[Solution] ```python
{}
```
"""

RAW_CASE_PROMPT = """Here are some relevant textual insights that can help you solve the machine learning task:
{} 
Now please solve the following machine learning task based on the textual insights above.
[Task]
{}
[train.py] ```python
{}
```
Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
```
"""

def get_task(task):
    # 读取 问题 & 对应的 basecode.py
    rp_path = RP_PATH.format(task)          # ./benchmarks/cirrhosis-outcomes/scripts/research_problem.txt
    python_path = PYTHON_PATH.format(task)  # ./benchmarks/cirrhosis-outcomes/env/train.py
    with open(rp_path) as file:
        rp = file.read()
    with open(python_path) as file:
        code = file.read()
    return rp, code

def get_case(task):
    # 读取 问题 & 对应的 basecode.py & 对应的 case.py
    # 调用 CASE_PROMPT.format(rp, code, case)
    rp_path = RP_PATH.format(task)
    python_path = PYTHON_PATH.format(task)
    case_path = CASE_PATH.format(task)
    with open(rp_path) as file:
        rp = file.read()
    with open(python_path) as file:
        code = file.read()
    with open(case_path)as file:
        case = file.read()
    return CASE_PROMPT.format(rp, code, case)

def get_prompt(task, context_num=0, strategy=None, raw=False):
    # raw: 是否使用 raw case, 
        # 如果 raw = True  => 调用 RAW_CASE_PROMPT.format(case, rp, code) => 从 heterogenous_similarity_ranking.json 中读取 [task] 所对应的 raw case
        # 如果 raw = False => 调用 get_case (i.e, CASE_PROMPT.format(rp, code, case))     => 则从 similarity_ranking.json 中读取 [task] 所对应的 case
    # context_num: 是否使用 few shot case, 
        # 如果 context_num = 0 => 调用 ZERO_SHOT_PROMPT.format(rp, code)          => 则不使用 few shot case
        # 如果 context_num > 0 => 调用 FEW_SHOT_PROMPT.format(examples, rp, code) => 则使用 few shot case
    
    rp, code = get_task(task)
    
    # Ablation Study
    if raw:
        with open("./config/heterogenous_similarity_ranking.json") as file:
            ranking_dictionary = json.load(file)
            case = ranking_dictionary[task]
            return RAW_CASE_PROMPT.format(case, rp, code)
    if context_num == 0:
        return ZERO_SHOT_PROMPT.format(rp, code)
    else:
        with open("./config/similarity_ranking.json") as file:
            ranking_dictionary = json.load(file)
        if strategy == "retrieval":
            selected_tasks = ranking_dictionary[task][:context_num] # 这里的 context_num 决定了 从 ranking_dictionary[task] 中取几个 case
        elif strategy == "random":
            selected_tasks = random.sample(ranking_dictionary[task], k=context_num)
        else:
            raise NotImplementedError("This strategy is not supported yet!")
        examples = ""
        for i in selected_tasks: # 遍历 selected_task 列表，把每个 task 都 append 到 get_case 
            examples += get_case(i)
        return FEW_SHOT_PROMPT.format(examples, rp, code)
        

if __name__ == '__main__':
    p = get_prompt("cirrhosis-outcomes", 
                    context_num=1,        # 是否使用 few shot case, 这里的 context_num 决定了 从 ranking_dictionary[task] 中取几个 case
                    strategy="retrieval", # 是否使用 retrieval case
                    raw=False             # 是否使用 raw case
                    )
    print(p)
