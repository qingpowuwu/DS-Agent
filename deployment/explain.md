## retrieval.py (用来生成./config/similarity_ranking.json)
总结: retrieval.py 文件通过遍历 DEPLOYMENT_TASKS 中的 task, 用 rb.retrieve_case 方法检索出每个 task 的相似 case, 并把结果写入 ./config/similarity_ranking.json 文件中

这段代码定义了一个名为 RetrievalDatabase 的类,用于构建和查询基于文本嵌入的案例检索数据库（self.embedding_bank）。该数据库可用于根据给定的查询文本检索相关的开发任务案例。代码的主要部分如下:

### 1. 初始化类 ```def __init__(self, model="BAAI/llm-embedder")```: 

主要是的目的是用来 生成变量 self.embedding_bank => 用于给 retrieve_case(self, query, num=12) 方法计算 similarity
 
1. 加载预训练的语言模型 self.model 和分词器 self.tokenizer。
2. 把 x_inputs(i.e, research problem) 做 embedding: i.e, 把 self.case_bank 中的每个 case 用 tokenizer 处理后，得到 x_inputs
3. 把 x_inputs 做 embedding: i.e, 把 self.case_bank 中的每个 case 用 tokenizer 处理后，得到 x_inputs
4. 用 self.model 预测 x_inputs 的输出 x_outputs : i.e, 把 input_ids 和 attention_mask 输入 self.model 得到 x_outputs => 嵌入向量(self.embedding_bank)

### 2. 检索相关案例 ```def retrieve_case(self, query, num=12)```:

主要的目的是：输入 query, 输出 1个 list, 这个 list 中含有和 query 相似的 12个 cases

1. retrieve_case 方法接受一个查询文本(query)和要检索的案例数量(num)。
2. 它计算查询文本的嵌入向量(x_embedding )。
3. 通过计算查询嵌入向量(x_embedding )与所有案例嵌入向量 (self.embedding_bank.T) 的相似度 (余弦相似度),获得相似度分数。
4. 根据相似度分数排序,返回最相关的案例任务名称列表。

### 3. 主程序入口:

1. 实例化 RetrievalDatabase 对象 rb。
2. 遍历部署任务列表 DEPLOYMENT_TASKS。
3. 对于每个部署任务:
  * 读取任务的研究问题描述文本(research_problem.txt)。
  * 使用 retrieve_case(query) 方法检索最相关的开发任务案例 ranking_dict[task]。

4. 最后,将 ranking_dict 写入 JSON 文件 ./config/similaity_ranking.json。

总的来说,这段代码构建了一个基于语义嵌入的案例检索系统。它利用预训练的语言模型来计算文本嵌入(self.embedding),并根据嵌入向量之间的相似度检索最相关的开发任务案例(similarity = (x_embedding @ self.embedding_bank.T).squeeze())。这种检索方法可以帮助找到与给定的部署任务最相关的开发案例,从而为部署任务提供有用的参考和指导。

生成的 similaity_ranking.json 文件包含了每个部署任务及其最相关的开发任务案例排名,可用于后续的分析和应用。

## execution.py

* execution_ori.py 就是原版的代码
* execution_print.py 和 execution.py 差不多就是加上了一些 print 的代码
* execution.py 是我修改后的代码，主要做了这几个修改：
  * 修改1: 加上了颜色，打印一些语句
  * 修改2: 增加了中间变量 script_path 
  * 修改3: 把 cmd 的打印语句变成 BLUE 了
    ```
        cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_name}"
        print(f'{BLUE}cmd = {cmd}{RESET}') # CUDA_VISIBLE_DEVICES=0 python -u prompt.py
    ```
  * 修改
* execution.py 的流程：
  * (1) 使用 subprocess.Popen 来执行 cmd 命令
  * (2) 使用 selectors 模块来监视脚本的标准输出和标准错误流,以实时捕获和打印输出。它还会将输出存储在 stdout_lines 和 stderr_lines 列表中。
  * (3) 在脚本执行完毕后,函数会检查返回码 return_code。
    * (3).1 如果 return_code 不为 0 (i.e, -1),则认为执行失败,将标准错误流的输出作为观察结果返回。
    * (3).2 如果 return_code 为 0,将标准输出流的输出作为观察结果返回。
    * (3).3 如果 observation 为 0 & 为 0,则认为只有标准错误流有输出,将标准错误流的输出作为观察结果返回。
  * (4) 如果
    * .py 文件可以运行: 返回 "The script has been executed. Here is the output:\n" + observation
    * .py 文件不能运行: 返回 Exception(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")

```
script_path = os.path.join(workspace_task_dir, script_name) # 想要执行的 ·python 脚本的地址
cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_name}" # 想要执行的 command line, ex: # CUDA_VISIBLE_DEVICES=0 python -u prompt.py
# (1) 使用 subprocess.Popen 来执行 cmd 命令
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=workspace_task_dir)
# (2) 使用 selectors 模块来监视脚本的标准输出和标准错误流,以实时捕获和打印输出。它还会将输出存储在 stdout_lines 和 stderr_lines 列表中。
stdout_lines = []
stderr_lines = []

selector = selectors.DefaultSelector()
selector.register(process.stdout, selectors.EVENT_READ)
selector.register(process.stderr, selectors.EVENT_READ)
...

# (3) 在脚本执行完毕后,函数会检查返回码 return_code。
if return_code != 0:
    observation = "".join(stderr_lines)
else:
    observation = "".join(stdout_lines)
if observation == "" and return_code == 0:
    # printed to stderr only
    observation = "".join(stderr_lines)

return "The script has been executed. Here is the output:\n" + observation
```

* 总结：execution.py 文件中定义了一个名为 execute_script 的函数,用于在指定的工作目录中执行一个 Python 脚本。
* execute_script 函数接受3个参数: script_name, workspace_task_dir, device
* 如果这个 .py 文件不存在,则会抛出异常 "The file {script_name} does not exist."
* 如果执行成功,则返回 "The script has been executed. Here is the output:\n" + observation
* 如果执行失败,则返回异常 "Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."


## evaluation.py

* evaluation_ori.py 就是最原版的代码，通过 运行代码 ```        if os.path.exists(f"./codes/{args.path}/{task}"):
            shutil.copytree(f"./codes/{args.path}/{task}", work_dir, symlinks=True, dirs_exist_ok=True)``` 把 用 ```generate_ori.py``` 和 gpt 对话得到的 .py 文件移动到 work_dir 目录下
  * 之后再运行 work_dir 目录下的 .py 文件
   
* evaluation_print.py 和 evaluation.py 基本一致，无非就是有更多的 print 语句

* evaluation.py 是我修改了之后的代码，因为我的 ```generate.py``` 把 和 gpt 对话得到的 .py 文件 & fitunelog_ansi 的文件放到一起了，所以这 evaluation.py 也在 evaluation_ori.py 的基础上作了一些修改:
  *  修改0：首先就是把一些我觉得有必要的地方打印了出来
  *  修改1：用.yaml 来更新参数
     ```
         # 修改1: 用 ./config/evaluate_py_config.yaml 来更新参数 ========================== 修改：开始 ==========================
         # Load config from config.yaml
         with open("./config/evaluate_py_config.yaml", "r") as f:
             config = yaml.safe_load(f)
     
         # Update args with config
         for key, value in config.items():
             if hasattr(args, key):
                 setattr(args, key, value)
         # 修改1: 用.yaml 来更新参数 ========================== 修改：结束 ==========================
     ```

  *  修改2: 因为 ./config/evaluate_py_config.yaml 里面的 task, 一般写成 task: ["smoker-status", "mohs-hardness", "bitcoin-price-prediction"] 的格式，所以 evaluation.py 里面的对应代码也需要更改:
    ```
       # 修改2: 修改代码来适配 .yaml 的 task 的参数 ==================================== 修改：开始 ====================================
       # Load tasks.
       if args.task == "all":
           tasks_to_evaluate = DEPLOYMENT_TASKS
       else:
           if isinstance(args.task, list):
               tasks = args.task
           else:
               tasks = args.task.split(',')
   
           tasks_to_evaluate = []
           for task in tasks:
               task = task.strip()
               if task in DEPLOYMENT_TASKS:
                   tasks_to_evaluate.append(task)
               else:
                   print(f"Warning: Task '{task}' is not in the list of deployment tasks. Skipping it.")
           if not tasks_to_evaluate:
               raise ValueError("No valid tasks found in the provided list.")
       print('tasks_to_evaluate = ', tasks_to_evaluate) # ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction']

    # 修改2: 修改代码来适配 .yaml 的 task 的参数 ==================================== 修改：结束 ====================================
    ```
* 修改3: 修改了各种文件夹的目录 => 来适配修改之后的 generate.py 生成的文件位置
  * work_dir -> workspace_task_dir, 并且改成 ``` workspace_task_dir = f"./1_workspace/{args.path}/{task}" ```
  * 增加了 benchmark_task_dir & benchmark_task_env_dir 当成中间变量
  * 增加了 task_pycode_dir 变量来当成中间变量, 并且改成 ``` task_pycode_dir = f"./codes/{args.path}/generated_codes/{task}" ```

* 总结：这个 evaluation.py 文件主要是 运行 generate.py 生成的和 gpt 交互得到的 训练 .py 文件，然后把训练结果保存到 result_csvfile_path = f"{result_dir}/{task}.csv"  文件中
  ```
          # (4) 创建结果目录 (result path)
        result_dir = f"./2_results/{args.path}"        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_csvfile_path = f"{result_dir}/{task}.csv" 
        print('\n')
        print('result_dir          =', result_dir)           # ./2_results/gpt-3.5-turbo-16k_False_0
        print('result_csvfile_path =', result_csvfile_path)  # ./2_results/gpt-3.5-turbo-16k_False_0/smoker-status.csv
  ```
    * (1) 从 submission.py 中找到 pattern
    * (2) 按照 pattern 来初始化 results, 然后不停的做实验, 并且把实验结果保存到 results 中
    * (3) 将结果保存到 result_csvfile_path 中
    * (4) 重复上述步骤 args.trials 次
    * (5) 重复上述步骤 args.task 次

## generate.py

* generate.py 文件主要是用来 和 gpt 交互 tasks_to_solve 次，并且重复对话 args.trails 次 (来保证 随机的鲁棒性)，每次 都调用 api max_num_try_calls 次 (来避免网络不稳定)

* generate_ori.py 原始的代码
* generate_print.py 修改后的代码，并且打印语句更多
* generate.py 修改后的代码，做了这几个修改
  * 修改1: 在保存的 fitunelog.ansi 文件 中加入计算的 tokens
    ```
    enc = tiktoken.get_encoding("cl100k_base") # 修改1: 利用这个 对象来计算 token 的数量
    ...
    def log_to_file(log_file_path, prompt, raw_completion, code_extracted):
    # log_file: 日志文件的 path
    # prompt: 提示
    # completion: 生成的代码
    """ Log the prompt and completion to a file. log_to_file()函数将提示(prompt)和生成的代码(completion)写入指定的日志文件,以供将来的微调(fine-tuning)使用。"""
    num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
    num_sample_tokens = len(enc.encode(raw_completion))
        
    ```  
  * 修改2: 在 parser 里面增加 gpt4_gpi 的 configuration
    ```
    # 修改2: ========================== 修改：开始 ==========================
    # (4) API Configuration
    parser.add_argument("--openai_api_key_path", default="./config/openai_api_key.txt")
    parser.add_argument("--openai_api_base", default="https://hk.xty.app/v1")
    # 修改2: ========================== 修改：结束 ==========================
    ```
 * 修改3: 可以用 ./config/generate_py_config.yaml 来更新参数
 * ```
    修改3: 用.yaml 来更新参数 ========================== 修改：开始 ==========================
    # Load config from config.yaml
    with open("./config/generate_py_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update args with config
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
   # 修改3: 用.yaml 来更新参数 ========================== 修改：结束 ==========================
  ```
* 修改4: 把变量 iteration -> 改成 num_try_call， 并且把 ```def generation(prompt, llm='gpt-3.5-turbo-16k', temperature=0.7, log_file=None):``` => 改成 => ```def generation(prompt, llm, temperature=0.7, log_file_path=None, max_num_try_calls=50):```
* 修改5: 修改了 def log_to_file 函数
  * 把 ```def log_to_file(log_file, prompt, completion)``` => 修改成 => ```def log_to_file(log_file_path, prompt, raw_completion, code_extracted):```
  * 增加了颜色
  * 写入了 prompt, completion => 修改成 => 保存了 prompt, raw_completion, code_extracted
  * 写入了 token 的长度
* 修改6: 修改代码来适配 .yaml 的 task 的参数
  * 因为我想要让 .yaml 文件里面的 task 为 类似 task: ["smoker-status", "mohs-hardness", "bitcoin-price-prediction"] 这样子的格式，所以对 generate_ori.py 文件也做了修改
  ```
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
```
* 修改7: 修改了 要保存的 .py 文件 & fitunelog.ansi 文件的路径
 ```
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
 ```

