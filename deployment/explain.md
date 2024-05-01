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
