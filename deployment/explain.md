## retrieval.py

这段代码定义了一个名为 RetrievalDatabase 的类,用于构建和查询基于文本嵌入的案例检索数据库。该数据库可用于根据给定的查询文本检索相关的开发任务案例。代码的主要部分如下:

### 1. 初始化类:

1. 加载预训练的语言模型和分词器 (Tokenizer)。
2. 定义用于表示查询和文档的提示 (Prompt)。
3. 从开发任务目录中读取所有案例文本,并构建提示。
4. 计算所有案例文本的嵌入向量,并保存在 embedding_bank 中。

### 2. 检索相关案例:

1. retrieve_case 方法接受一个查询文本和要检索的案例数量。
2. 它计算查询文本的嵌入向量。
3. 通过计算查询嵌入向量与所有案例嵌入向量的相似度 (余弦相似度),获得相似度分数。
4. 根据相似度分数排序,返回最相关的案例任务名称列表。

### 3. 主程序入口:

1. 实例化 RetrievalDatabase 对象。
2. 遍历部署任务列表 DEPLOYMENT_TASKS。
3. 对于每个部署任务:
  * 读取任务的研究问题描述文本。
  * 使用 retrieve_case 方法检索最相关的开发任务案例。
  * 将任务名称和检索到的相关案例列表存储在字典 ranking_dict 中。

4. 最后,将 ranking_dict 写入 JSON 文件 config/similaity_ranking.json。

总的来说,这段代码构建了一个基于语义嵌入的案例检索系统。它利用预训练的语言模型来计算文本嵌入,并根据嵌入向量之间的相似度检索最相关的开发任务案例。这种检索方法可以帮助找到与给定的部署任务最相关的开发案例,从而为部署任务提供有用的参考和指导。

生成的 similaity_ranking.json 文件包含了每个部署任务及其最相关的开发任务案例排名,可用于后续的分析和应用。