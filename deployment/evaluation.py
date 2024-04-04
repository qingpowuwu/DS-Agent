import os
import shutil
import argparse
import numpy as np
import pandas as pd
from execution import execute_script

DEPLOYMENT_TASKS = ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat', 'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews', 'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1', 'uwave-gesture-library', 'traffic', 'boolq', 'crab-age', 'concrete-strength', 'jigsaw']

def get_args():
    parser = argparse.ArgumentParser()
    # Model Information
    parser.add_argument("--path", default="gpt-3.5-turbo-16k_False_0")		# Code path
    parser.add_argument("--task", default="electricity")		                    # Task name
    # Generation Configuration
    parser.add_argument("--trials", default=10, type=int)    		                # Number of trials (fixed)
    # Device info
    parser.add_argument("--device", default="0", type=str)    		                # Device num
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    # Load tasks.
    if args.task == "all":
        tasks_to_evaluate = DEPLOYMENT_TASKS
    else:
        assert args.task in DEPLOYMENT_TASKS
        tasks_to_evaluate = [args.task]
    
    # Evaluate all the tasks.
    for task in tasks_to_evaluate:
        
        # Create a workspace (当前的工作目录，我们做的实验都会放到这个目录里面)
        work_dir = f"./workspace/{args.path}/{task}"  # ./workspace/gpt-3.5-turbo-16k_False_0/electricity
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            
        # ============= 将指定任务的环境文件 (xxx/benchmarks/electricity/env ) & 代码文件复制到 工作目录 (work_dir) 中 ==============
        # (1)将指定任务的环境文件 (xxx/benchmarks/electricity/env ) 到 工作目录 (work_dir) 中
        if os.path.exists(f"../development/MLAgentBench/benchmarks/{task}"):
            shutil.copytree(f"../development/MLAgentBench/benchmarks/{task}/env", work_dir, symlinks=True, dirs_exist_ok=True)
            
        # (2)将 generate.py 生成的代码文件 (xxx/codes/gpt-3.5-turbo-16k_False_0/electricity) 复制到 工作目录 (work_dir) 中
        if os.path.exists(f"./codes/{args.path}/{task}"):
            shutil.copytree(f"./codes/{args.path}/{task}", work_dir, symlinks=True, dirs_exist_ok=True)
         # ============= 将指定任务的环境文件 (xxx/benchmarks/electricity/env ) & 代码文件复制到 工作目录 (work_dir) 中 ==============
        
        # Find submission pattern
        line = None
        with open(f"{work_dir}/submission.py") as file:
            for line in file:
                if "print" in line:
                    pattern = line
        assert line
        
        if "MSE" in line and "MAE" in line:
            results = [[], []]
        else:
            results = []
        
        pattern = line.split("\"")[1].split(":")[0]
        
        # 创建结果目录 (result path)
        result_dir = f"results/{args.path}"          # xxx/results/gpt-3.5-turbo-16k_False_0
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_filename = f"{result_dir}/{task}.csv" # xxx/results/gpt-3.5-turbo-16k_False_0/electricity.csv

        # 遍历 args.trials => 开始不停的做实验, 对于每次试验:
        # (1) 执行训练脚本。
        # (2) 从输出中提取结果,如 MSE(均方误差)和 MAE(平均绝对误差)。
        for idx in range(args.trials): 
            filename = f"train_{idx}.py" # train_1.py (之前用 generate.py 生成的)
            
            # 运行 xxx.py 代码，并且输出日志（log）
            log = execute_script(filename, work_dir=work_dir, device=args.device)
            
             # 根据提交代码中的模式(pattern),从输出脚本(log)中提取结果,如 MSE 和 MAE。如果无法提取结果,则将结果设置为 -1.0,表示执行失败。
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
                    
        # 如果有 MSE 和 MAE 两个指标,则将它们作为两列保存;否则,只保存一列结果。
        if "MSE" in line and "MAE" in line:
            results = pd.DataFrame(results)
        else:
            results = pd.DataFrame(results).transpose()
            
        # 将结果保存为 CSV 文件。
        results.to_csv(result_filename, index=False, header=False) 
        print("results")
        print("="*100)
            
        
        
