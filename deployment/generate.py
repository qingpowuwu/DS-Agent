

import openai
import argparse
import torch
import os
import anthropic
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import get_prompt
import yaml

# 定义颜色代码
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"  # 新增黄色代码
GREEN = "\033[92m"   # 新增绿色代码
RESET = "\033[0m"

print(f'\n{YELLOW}==================================================================== generate.py: 开始 ==================================================================={RESET}\n')

# openai.api_key = "FILL IN YOUR KEY HERE."
# openai.api_base = "http://localhost:8000/v1"

enc = tiktoken.get_encoding("cl100k_base") # 修改1: 利用这个 对象来计算 token 的数量

DEVELOPMENT_TASKS = ["feedback", "airline-reviews", "textual-entailment", "chatgpt-prompt", "ett-m2", "ili", "handwriting", "ethanol-concentration", "media-campaign-cost", "wild-blueberry-yield", "spaceship-titanic", "enzyme-substrate"]
DEPLOYMENT_TASKS = ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat', 'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews', 'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1', 'uwave-gesture-library', 'traffic', 'boolq', 'crab-age', 'concrete-strength', 'jigsaw']

#%%

def get_args():
    parser = argparse.ArgumentParser()
    # (1) Model Information
    parser.add_argument("--llm",  default="gpt-3.5-turbo-16k")		# LLM name
    parser.add_argument("--task", default=['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction'])   # ML Task name
    # (2) Context Configuration
    parser.add_argument("--shot", default=0, type=int)              # Number of examples in context (上下文中的例子数量)
    parser.add_argument("--retrieval", default=False,               # Whether activate retrieval (是否使用检索)
                        action='store_true')
    parser.add_argument("--raw", default=False,                     # Whether use raw cases (是否使用原始案例)
                        action='store_true')
    # (3) Generation Configuration
    parser.add_argument("--temperature", default=0.7, type=float)   # Temperature (fixed)
    parser.add_argument("--trials", default=10, type=int)    		# Number of trials (fixed) (试验次数)
    # 修改2: ========================== 修改：开始 ==========================
    # (4) API Configuration
    parser.add_argument("--openai_api_key_path", default="./config/openai_api_key.txt")
    parser.add_argument("--openai_api_base", default="https://hk.xty.app/v1")
    # 修改2: ========================== 修改：结束 ==========================

    args = parser.parse_args()
    
    # 修改3: 用.yaml 来更新参数 ========================== 修改：开始 ==========================
    # Load config from config.yaml
    with open("./config/generate_py_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update args with config
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    # 修改3: 用.yaml 来更新参数 ========================== 修改：结束 ==========================
    return args

def generation(prompt, llm, temperature=0.7, log_file_path=None, max_num_try_calls=50):
    """_summary_: generation()函数是核心函数,用于通过OpenAI的API生成代码
        接受一个提示(prompt)、语言模型名称(llm)、温度(temperature)和日志文件路径(log_file_path)作为参数。
            它会尝试多次(最多50次)调用API,直到成功生成代码 or 达到最大尝试次数。
        生成的代码会打印到控制台,并将提示和生成的代码写入日志文件。

    Args:
        - prompt     : (_type_)         : _description_
        - llm        : (_type_)         : _description_
        - temperature: (float, optional): _description_.
                                                    [Defaults to 0.7]
        - log_file: (_type_, optional): _description_.
                                                    [Defaults to None]

    Returns:
        - : (_type_): _description_
    """
    print('\n\032[94m-------------------------------------------------- generation(prompt, llm, temperature=0.7, log_file=None) 函数: 开始 --------------------------------------------------\032[0m\n')

    raw_request = {
        "model": llm,
        "temperature": temperature,
        "max_tokens": 1500,
        "stop": [] or None,
    }
    print('raw_request = ', raw_request) # {'model': 'gpt-3.5-turbo-16k', 'temperature': 0.7, 'max_tokens': 1500, 'stop': None}
    num_try_call = 0
    code_extracted = None
    while num_try_call < max_num_try_calls: # 尝试最多50次 调用API (这里的 num_teration 主要是为了防止 API 调用失败 (例如网络失败，并不是 chatgpt 里面的新的对话), 一直重试) 
        print(f'\n{YELLOW}num_try_call = {YELLOW}{num_try_call}{RESET}\n') # 0
        try:
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(**{"messages": messages,**raw_request}) # <class 'openai.openai_object.ChatCompletion'>
            raw_completion = response["choices"][0]["message"]["content"]         # llm 返回的原始 completion
            code_extracted = raw_completion.split("```python")[1].split("```")[0] # 提取生成的代码
            print(f'{RED}[[[prompt]]]           \n {RED}{prompt}{RESET}')
            print(f'{GREEN}[[[raw_completion]]] \n {raw_completion}{RESET}')
            print(f'{BLUE}[[[code_extracted]]]  \n {code_extracted}{RESET}') 
            if not code_extracted.strip(" \n"):
                continue
            break
        except Exception as e: # 尝试调用 gpt 的 API, 用来解决网络不稳定的问题
            num_try_call += 1
            print(f"===== 重试: num_try_call: {num_try_call} =====")
            print(f"Error occurs when calling API: {e}")
        continue
    if not code_extracted:
        code_extracted = ""
    
    print(f'{BLUE} 已经 遍历了所有的 num_try_call: 最终的 [[[code_extracted]]]  \n {code_extracted}{RESET}') 
    log_to_file(log_file_path, prompt, raw_completion, code_extracted) # 将 prompt 和 completion 和 code_extracted 写入日志文件
    
    print('\n\032[94m-------------------------------------------------- generation(prompt, llm, temperature=0.7, log_file=None) 函数: 结束 --------------------------------------------------\032[0m\n')

    return code_extracted
    
# 修改4: 计算 tokens ==================================== 修改：开始 ====================================
def log_to_file(log_file_path, prompt, raw_completion, code_extracted):
    # log_file: 日志文件的 path
    # prompt: 提示
    # completion: 生成的代码
    """ Log the prompt and completion to a file. log_to_file()函数将提示(prompt)和生成的代码(completion)写入指定的日志文件,以供将来的微调(fine-tuning)使用。"""
    num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
    num_sample_tokens = len(enc.encode(raw_completion))
        
    # 直接根据 prompt 和 completion 的类型, 获取它们的名称 (这样子是为了方便后续的日志查看)
    # prompt_var_name = prompt.__class__.__name__
    # completion_var_name = raw_completion.__class__.__name__
    # code_extracted_var_name = code_extracted.__class__.__name__

    # Logging .txt 文件 for finetuning
    with open(log_file_path, "wt") as f:
        f.write("\n[This is a split string for prompt]\n")
        f.write(f'{RED}[[[prompt]]]       \n {RED}{prompt}{RESET}\n')
        
        f.write("\n[This is a split string for finetuning]\n")
        f.write(f'{GREEN}[[[raw_completion]]]  \n {GREEN}{raw_completion}{RESET}\n')
        
        f.write(f'{BLUE}[[[code_extracted]]]  \n {BLUE}{code_extracted}{RESET}\n')
        
        f.write("\n[This is a split string for counting tokens]\n")
        f.write(f"{YELLOW}Prompt: {YELLOW}{num_prompt_tokens}, {YELLOW}Completion: {YELLOW}{num_sample_tokens}")

# 修改4: 计算 tokens ==================================== 修改：结束 ====================================

if __name__ == '__main__':
    args = get_args()

    # (0).1 Set OpenAI API Key
    openai.api_key = open(args.openai_api_key_path).read() # open("./config/openai_api_key.txt").read()
    os.environ["OPENAI_API_KEY"] = openai.api_key 
    openai.api_base = args.openai_api_base                 # "https://hk.xty.app/v1"

    print('\nargs = ', args) # Namespace(llm='gpt-3.5-turbo-16k', task='detect-ai-generation', shot=0, retrieval=False, raw=False, temperature=0.7, trials=10)
    print('openai.api_key  = ', openai.api_key)  # DS_Agent_repository/DS-Agent/deployment/generate.py
    print('openai.api_base = ', openai.api_base) # https://hk.xty.app/v1
    
    # 修改5: 修改代码来适配 .yaml 的 task 的参数 ==================================== 修改：开始 ====================================
    # (0).2 Load Tasks
    if args.task == "all":
        tasks_to_solve = DEPLOYMENT_TASKS    # 把 list 中的所有 tasks 都放到 tasks_to_solve 变量中
    else:
        if isinstance(args.task, list):
            tasks = args.task
        else:
            tasks = args.task.split(',')

        tasks_to_solve = []
        for task in tasks:
            task = task.strip()
            if task in DEPLOYMENT_TASKS:
                tasks_to_solve.append(task)
            else:
                print(f"Warning: Task '{task}' is not in the list of deployment tasks. Skipping it.")
        if not tasks_to_solve:
            raise ValueError("No valid tasks found in the provided list.")
    print('tasks_to_solve = ', tasks_to_solve) # ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction']
    # 修改5: 修改代码来适配 .yaml 的 task 的参数 ==================================== 修改：结束 ====================================
        
    # (0).3 设定要保存的路径: 前缀 prefix 代表了当前实验 LLM 名称、检索策略和 shot 数量
    prefix = f"{args.llm}_{args.retrieval}_{args.shot}" if not args.raw else f"{args.llm}_{args.retrieval}_{args.shot}_raw"
    print('prefix             = ', prefix) # gpt-3.5-turbo-16k_True_1
    
    # (1) Create the path for gpt 生成的 py 代码: response
    generate_codes_dir = f"./codes/{prefix}/generated_codes" 
    print('generate_codes_dir = ', generate_codes_dir) # ./codes/gpt-3.5-turbo-16k_False_0/generated_codes
    if not os.path.exists(generate_codes_dir):
        os.makedirs(generate_codes_dir)
       
    # (2) Create Finetune Logs ansi: prompt 和 response
    finetune_log_ansi_dir = f"./codes/{prefix}/finetune_log_ansi"
    print('finetune_log_ansi_dir       = ', finetune_log_ansi_dir) # ./codes/gpt-3.5-turbo-16k_False_0/finetune_log_ansi
    if not os.path.exists(finetune_log_ansi_dir):
        os.makedirs(finetune_log_ansi_dir)
        
    print('\n-------------------------------- 开始遍历所有的 tasks --------------------------------\n')
    # 遍历所有的 tasks，通过 generation() 函数生成代码，并保存到指定的文件中
    task_count = 0
    for task in tasks_to_solve:
        # (1) 创建任务文件夹: i.e, response
        temp_generatedcode_dir = f"{generate_codes_dir}/{task}"    # 当前 llm/task 的路径
        print('temp_generatedcode_dir = ', temp_generatedcode_dir) # ./codes/gpt-3.5-turbo-16k_False_0/generated_codes/detect-ai-generation
        # (2) 创建微调日志文件夹: i.e, prompt 和 response
        temp_finetunedir       = f"{finetune_log_ansi_dir}/{task}"
        print('temp_finetunedir       = ', temp_finetunedir)       # ./codes/gpt-3.5-turbo-16k_False_0/finetune_log_ansi/detect-ai-generation
        if not os.path.exists(temp_generatedcode_dir):
            os.makedirs(temp_generatedcode_dir)
        if not os.path.exists(temp_finetunedir):
            os.makedirs(temp_finetunedir)
            

        print(f"{YELLOW}task          = {YELLOW}{task}{RESET}")     # detect-ai-generation
        print('temp_generatedcode_dir = ', temp_generatedcode_dir)  # ./codes/gpt-3.5-turbo-16k_True_1/detect-ai-generation
        print('temp_finetunedir       = ', temp_finetunedir)        # ./codes/gpt-3.5-turbo-16k_True_1/finetune_log_ansi/detect-ai-generation

        print('\n-------------------------------- 开始遍历所有的 trials --------------------------------\n')

        for idx in range(args.trials): # 这里的 trial 指的是 重复对话的次数, 目的是 重复实验
            print(f'\n{YELLOW}-------- task: {YELLOW}{task} ----- {task_count}/{len(tasks_to_solve)} -------------------- 正在遍历 trials, 当前为重复第几次对话[idx] = {idx}/{args.trials} ----------------------------{RESET}\n')
            log_file_path_log = f"{temp_finetunedir}/{idx}.ansi"
            print('log_file_path = ', log_file_path_log) # ./codes/gpt-3.5-turbo-16k_True_1/finetune_log_ansi/detect-ai-generation/0.txt
            prompt   = get_prompt(task, 
                                  context_num=args.shot, 
                                  strategy="retrieval" if args.retrieval else "random", 
                                  raw=args.raw
                                  )
            code_extracted = generation(prompt, 
                                  args.llm, 
                                  temperature=args.temperature, 
                                  log_file_path=log_file_path_log, # ./codes/gpt-3.5-turbo-16k_True_1/finetune_log_ansi/detect-ai-generation/0.txt
                                  max_num_try_calls=50
                                  )
            pycode_extracted_path = f"{temp_generatedcode_dir}/train_{idx}.py"
            print('filename = ', pycode_extracted_path) # ./codes/gpt-3.5-turbo-16k_False_0/detect-ai-generation/train_0.py
            with open(pycode_extracted_path, "wt") as file: # 把提取出来的 python 代码保存到 pycode_extracted_path 里面
                file.write(code_extracted)
            
        task_count = task_count + 1

print(f'\n{YELLOW}==================================================================== generate.py: 结束 ==================================================================={RESET}\n')

# 我个人的理解是, 这个 generate.py 脚本的主要作用是: 首先确定要处理的 tasks:
# 开始遍历 tasks_to_solve 中的所有 task:
    # 1. 为当前 task 创建一个路径, 用于保存生成的代码 (temp_generatedcode_dir)
    # 2. 为当前 task 创建一个路径, 用于保存微调日志 (temp_finetunedir)
    # 3. 开始遍历 trials, 也就是对话的轮数:
        # 通过 调用 prompt.py 中的 get_prompt() 函数, 生成一个 prompt, 
        # 然后把 prompt => 输入 => generation() 函数 => 用 OpenAI 的 API => 生成回答 => 提取代码 (code_generated), 
        # 并将生成的代码 (code_generated) 保存到指定的文件 (codes/generated_codes/{task}/{trials})。

# 这个 generate.py 的目的是为了生成代码, 用于后续的微调(fine-tuning)。
# 这个 generate_print.py 主要是对 generate.py 的代码 保存的3个文件目录修改了一下，把它们放到了同1个目录:

# generate.py
    # temp_pathname = f"{pathname}/{task}"                       # ./codes/gpt-3.5-turbo-16k_False_0/detect-ai-generation
        # 这个路径下生成一个 train_{idx}.py 文件,文件内容是 response
        # train_1.py 文件内容是 response
        # train_2.py 文件内容是 response
        # ...
    # temp_fineturnedir = f"{finetune_dir}/{task}" # ./codes/gpt-3.5-turbo-16k_False_0/finetune_log_ansi/detect-ai-generation
        # 这个路径下生成一个 {idx}.txt 文件,文件内容是 prompt 和 response
        # 1.txt 文件内容是 prompt 和 response
        # 2.txt 文件内容是 prompt 和 response
        # ...
    # log_folder = f"./log_prompt_response/{task}" # ./log_prompt_response/detect-ai-generation
        # 这个路径下生成一个 {idx}.log 文件,文件内容是 task, idx, prompt 和 response
        # 1.log 文件内容是 task, idx, prompt 和 response
        # 2.log 文件内容是 task, idx, prompt 和 response
        # ...
         
# generate_print.py
    # (1) 创建任务文件夹: i.e, response
    # temp_generatedcode_dir = f"{generate_codes_dir}/{task}"        # ./codes/gpt-3.5-turbo-16k_False_0/generated_codes/detect-ai-generation
        # 这个路径下生成一个 train_{idx}.py 文件,文件内容是 response
        # train_1.py 文件内容是 response
        # train_2.py 文件内容是 response
        # ...
    # (2) 创建微调日志文件夹: i.e, prompt 和 response
    # temp_finetunedir       = f"{finetune_dir}/{task}"              # ./codes/gpt-3.5-turbo-16k_False_0/finetune_log_ansi/detect-ai-generation
        # 这个路径下生成一个 {idx}.txt 文件,文件内容是 prompt 和 response
        # 1.txt 文件内容是 prompt 和 response
        # 2.txt 文件内容是 prompt 和 response
        # ...
    # log_folder = f"./log_prompt_response/{task}" # ./log_prompt_response/detect-ai-generation
        # 这个路径下生成一个 {idx}.log 文件,文件内容是 task, idx, prompt 和 response
        # 1.log 文件内容是 task, idx, prompt 和 response
        # 2.log 文件内容是 task, idx, prompt 和 response
        # ...
