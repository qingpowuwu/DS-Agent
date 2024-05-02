




# 第一次快速的测试：用 openai 的 api 来做 cifar10 task

python -u -m MLAgentBench.runner --python $(which python) --task cifar10 --device 0 --log-dir first_test  --work-dir workspace --llm-name gpt-4 --edit-script-llm-name gpt-4 --fast-llm-name gpt-3.5-turbo >  first_test/log 2>&1


# 做评估

"""
python -m MLAgentBench.eval --log-folder final_exp_logs/claude-v1/2023-04-15_12-00-00 --task cifar10 --output-file cifar10_claude_eval.json
"""

该命令是用于评估 MLAgentBench benchmark-test 中claude-v1模型在 cifar10任务上的性能,并将评估结果保存到cifar10_claude_eval.json文件中。让我们分解一下这条命令:

1. python -m MLAgentBench.eval

    * 这部分告诉Python从MLAgentBench模块的eval.py文件中运行代码。eval.py文件包含了评估代理模型性能的代码。
--log-folder final_exp_logs/claude-v1/2023-04-15_12-00-00

2. --log-folder参数指定了包含代理模型运行日志的文件夹路径。
    * final_exp_logs/claude-v1/2023-04-15_12-00-00是一个具体的路径,表示claude-v1模型在2023年4月15日12:00:00运行的日志所在文件夹。
3. --task cifar10

    * --task参数指定了要评估的任务名称,这里是cifar10任务。
4. --output-file cifar10_claude_eval.json

    * --output-file参数指定了评估结果输出的JSON文件名称和路径。
    * cifar10_claude_eval.json表示输出文件将被命名为cifar10_claude_eval.json,并保存在当前工作目录下。

因此,该命令的作用是:

1. 从指定的日志文件夹final_exp_logs/claude-v1/2023-04-15_12-00-00中读取claude-v1模型在cifar10任务上的运行日志。
2. 根据这些日志,对claude-v1模型在cifar10任务上的性能进行评估,计算诸如成功率、相对于基线的平均改进等指标。
3. 将评估结果以JSON格式保存到cifar10_claude_eval.json文件中。

这个评估的结果可以用于后续分析不同模型在不同任务上的表现,或者与基线模型进行比较等。通过这种方式,我们可以系统地评测和比较各种大型语言模型在MLAgentBench基准测试中的能力。

# 准备数据集: 调用 prepare_task.py

'''
python -u -m MLAgentBench.prepare_task <task_name> $(which python)
'''

在 /data/Project_3_Science_Agent/1_Agents_folders/MLAgentBench 目录下

'''
python -u -m MLAgentBench.prepare_task cifar10 $(which python)
'''

在 /data/Project_3_Science_Agent/1_Agents_folders/MLAgentBench/MLAgentBench 目录下

'''
python prepare_task.py cifar10 $(which python)

python prepare_task.py imdb $(which python)

python prepare_task.py feedback $(which python)

python prepare_task_print.py feedback $(which python)

![alt text](MLAgentBench/benchmarks/feedback-dataset截图.png)

python prepare_task_print.py spaceship-titanic $(which python)
![alt text](MLAgentBench/benchmarks/spaceship-titanic-截图1.png)
![alt text](MLAgentBench/benchmarks/spaceship-titanic-截图2.png)



python prepare_task_print.py vectorization $(which python)

![alt text](MLAgentBench/benchmarks/vectorization-截图.png)

'''

# 开始运行实验: 调用 runner.py

'''
python -u -m MLAgentBench.runner --python $(which python) --task cifar10 --device 0 --log-dir first_test  --work-dir workspace --llm-name gpt-4 --edit-script-llm-name gpt-4 --fast-llm-name gpt-3.5-turbo >  first_test/log 2>&1
'''

在 /data/Project_3_Science_Agent/1_Agents_folders/MLAgentBench/MLAgentBench 目录下

'''
python runner.py --python $(which python) --task cifar10 --device 0 --log-dir first_test  --work-dir workspace --llm-name gpt-4 --edit-script-llm-name gpt-4 --fast-llm-name gpt-3.5-turbo
'''

'''
python MLAgentBench.runner --python $(which python) --task cifar10 --device 0 --log-dir first_test  --work-dir workspace --llm-name gpt-4 --edit-script-llm-name gpt-4 --fast-llm-name gpt-3.5-turbo
'''

# 准备数据集