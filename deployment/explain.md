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

## execution.py

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

总的来说,这段代码构建了一个基于语义嵌入的案例检索系统。它利用预训练的语言模型来计算文本嵌入(self.embedding),并根据嵌入向量之间的相似度检索最相关的开发任务案例(similarity = (x_embedding @ self.embedding_bank.T).squeeze())。这种检索方法可以帮助找到与给定的部署任务最相关的开发案例,从而为部署任务提供有用的参考和指导。

生成的 similaity_ranking.json 文件包含了每个部署任务及其最相关的开发任务案例排名,可用于后续的分析和应用。
