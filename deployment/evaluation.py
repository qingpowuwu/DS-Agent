import os
import shutil
import argparse
import numpy as np
import pandas as pd
from execution import execute_script
import yaml

# 定义颜色代码
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"  # 新增黄色代码
GREEN = "\033[92m"   # 新增绿色代码
RESET = "\033[0m"

print(f'\n{YELLOW}==================================================================== evaluation_print.py: 开始 ==================================================================={RESET}\n')


DEPLOYMENT_TASKS = ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat', 'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews', 'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1', 'uwave-gesture-library', 'traffic', 'boolq', 'crab-age', 'concrete-strength', 'jigsaw']

#%%

def get_args():
    parser = argparse.ArgumentParser()
    # (1) Model Information
    parser.add_argument("--path", default="gpt-3.5-turbo-16k_False_0")		        # Code path
    parser.add_argument("--task", default=['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction'])		                    # Task name

    # (2) Generation Configuration
    parser.add_argument("--trials", default=10, type=int)    		                # Number of trials (fixed)

    # (3) Device info
    parser.add_argument("--device", default="0", type=str)    		                # Device num
    args = parser.parse_args()
    
    # 修改1: 用.yaml 来更新参数 ========================== 修改：开始 ==========================
    # Load config from config.yaml
    with open("./config/evaluate_py_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update args with config
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    # 修改1: 用.yaml 来更新参数 ========================== 修改：结束 ==========================
    return args



if __name__ == '__main__':
    args = get_args()
    
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
    # Evaluate all the tasks.
    for task in tasks_to_evaluate: # ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction']
        
        # (0) 创建 1个 workspace (i.e, 当前的工作目录，我们这个 evaluation.py 做的实验都会放到这个目录里面)
        workspace_task_dir = f"./1_workspace/{args.path}/{task}" 
        if not os.path.exists(workspace_task_dir):
            os.makedirs(workspace_task_dir)
            
        # (1)将指定任务的环境文件夹 (xxx/development/MLAgentBench/benchmarks/electricity/env) 中的所有文件 => 复制到 => 工作目录 (workspace_task_dir) 中
        benchmark_task_dir = f"../development/MLAgentBench/benchmarks/{task}"
        benchmark_task_env_dir = f"../development/MLAgentBench/benchmarks/{task}/env"
        if os.path.exists(benchmark_task_env_dir):
            # 把 benchmark_task_env_dir 文件夹中的所有文件 复制到 workspace_task_dir 文件夹中
            shutil.copytree(benchmark_task_env_dir, # ../development/MLAgentBench/benchmarks/detect-ai-generation/env
                            workspace_task_dir,     # ./1_workspace/gpt-3.5-turbo-16k_False_0/detect-ai-generation
                            symlinks=True, 
                            dirs_exist_ok=True
                            )
            
            
        # (2)将 generate.py 生成的代码文件 (xxx/codes/gpt-3.5-turbo-16k_False_0/electricity) 复制到 工作目录 (work_dir) 中
        task_pycode_dir = f"./codes/{args.path}/generated_codes/{task}"
        if os.path.exists(task_pycode_dir):
            shutil.copytree(task_pycode_dir, workspace_task_dir, symlinks=True, dirs_exist_ok=True)
         # ============= 将指定任务的环境文件 (xxx/benchmarks/electricity/env ) & 代码文件复制到 工作目录 (work_dir) 中 ==============
        
        print(f'\n{YELLOW}[task = {task}]{RESET}\n')
        print('workspace_task_dir       =', workspace_task_dir)   # ./workspace/gpt-3.5-turbo-16k_False_0/detect-ai-generation
        print('benchmark_task_dir       =', benchmark_task_dir)    # ../development/MLAgentBench/benchmarks/detect-ai-generation
        print('benchmark_task_env_dir   =', benchmark_task_env_dir)  # ../development/MLAgentBench/benchmarks/detect-ai-generation/env
        print('task_pycode_dir            =', task_pycode_dir)          # ./codes/gpt-3.5-turbo-16k_False_0/detect-ai-generation
        print('os.path.exists(workspace_task_dir)     =', os.path.exists(workspace_task_dir)) # False
        print('os.path.exists(benchmark_task_env_dir) =', os.path.exists(benchmark_task_env_dir)) # True
        print('os.path.exists(task_pycode_dir)          =', os.path.exists(task_pycode_dir)) # True

        print('\n')

        # (3) 从 submission.py 文件中 找到 submission pattern
        line = None
        with open(f"{workspace_task_dir}/submission.py") as file:
            for line in file:
                if "print" in line: # 如果 line 中包含 "print" 字符串
                    pattern = line
        assert line
        
        print('pattern =', pattern) # pattern = print("final Accuracy on test set: ", rmse)
        
        # 根据 submission.py 中的 print 语句,确定 初始化 results 的格式, 这个 results 保存了每1次 trial 实验的结果, 例如 [0.9902176523761718, 0.9902176523761718, 0.9902176523761718, 0.9902176523761718]
        if "MSE" in line and "MAE" in line:
            results = [[], []]
        else:
            results = []
        
        pattern = line.split("\"")[1].split(":")[0] 
        
        print('pattern =', pattern) # Final AUROC on test set
        print('results =', results) # []
        
        # (4) 创建结果目录 (result path)
        result_dir = f"./2_results/{args.path}"        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_csvfile_path = f"{result_dir}/{task}.csv" 
        print('\n')
        print('result_dir          =', result_dir)           # ./2_results/gpt-3.5-turbo-16k_False_0
        print('result_csvfile_path =', result_csvfile_path)  # ./2_results/gpt-3.5-turbo-16k_False_0/smoker-status.csv

        # (5) 遍历 args.trials => 开始不停的做实验, 对于每次试验:
        # (5).1 执行训练脚本。
        # (5).2 从输出中提取结果,如 MSE(均方误差)和 MAE(平均绝对误差)。
        print(f'\n{YELLOW}===================== 开始遍历 args.trials => 开始不停的做实验 ====================={RESET}')
        for idx in range(args.trials): 
            pycode_file_name = f"train_{idx}.py" # train_1.py (之前用 generate.py 生成的)
            print('\n')
            print(f'{YELLOW}idx = {idx}/{args.trials}, pycode_file_name = {pycode_file_name}{RESET}\n') # 0/5
            print('os.path.exists(script_path = os.path.join(workspace_task_dir, script_name)) = ', os.path.exists(os.path.join(workspace_task_dir, pycode_file_name))) # True
            # print('os.path.exists(pycode_file_path) =', os.path.exists(pycode_file_name)) # True
            
            # (5).1 运行 train_1.py 代码，并且输出日志（log）
            log = execute_script(pycode_file_name, workspace_task_dir=workspace_task_dir, device=args.device)

            print('log = ', log) # The script has been executed. Here is the output:
                                    # Final area under the ROC curve (AUROC) on validation set: 0.8563294523983053
                                    # Final AUROC on test set: 0.9902176523761718.
            
             # (5).2 根据提交代码中的模式(pattern),从输出脚本(log)中提取结果,如 MSE 和 MAE。如果无法提取结果,则将结果设置为 -1.0,表示执行失败。
            if pattern in log:
                if "MSE" in line and "MAE" in line:
                    results[0].append(float(log.split(pattern)[1].split(":")[1].split(",")[0]))
                    results[1].append(float(log.split(pattern)[1].split(":")[2].strip(",.\n ")))
                else:
                    results.append(float(log.split(pattern)[1].split(":")[1].strip(",.\n ")))
            else:      # Fail to execute
                if "MSE" in line and "MAE" in line:
                    results[0].append(-1.0)
                    results[1].append(-1.0)
                else:
                    results.append(-1.0)

            print('results =', results)  # [0.9902176523761718, 0.9902176523761718]
                    
        # 如果有 MSE 和 MAE 两个指标,则将它们作为两列保存;否则,只保存一列结果。
        if "MSE" in line and "MAE" in line:
            results = pd.DataFrame(results)
        else:
            results = pd.DataFrame(results).transpose()
            
        # 将结果保存为 CSV 文件。
        results.to_csv(result_csvfile_path, index=False, header=False) 
        print("results")
        print("="*100)
            
        
        
print(f'\n{YELLOW}==================================================================== evaluation_print.py: 结束 ==================================================================={RESET}\n')

# 总结：这个 evaluation.py 脚本的作用是:
# (1) 从 submission.py 中找到 pattern
# (2) 按照 pattern 来初始化 results, 然后不停的做实验, 并且把实验结果保存到 results 中
# (3) 将结果保存到 result_csvfile_path 中
# (4) 重复上述步骤 args.trials 次
# (5) 重复上述步骤 args.task 次

