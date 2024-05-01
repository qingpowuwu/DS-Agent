import os
import subprocess
import selectors

# 定义颜色代码
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"  # 新增黄色代码
GREEN = "\033[92m"   # 新增绿色代码
RESET = "\033[0m"

print(f'\n{YELLOW}==================================================================== execution.py: 开始 ==================================================================={RESET}\n')

# 这段代码定义了一个名为 execute_script 的函数,用于在指定的工作目录中执行一个 Python 脚本。
def execute_script(script_name, workspace_task_dir = ".", device="0"):
    """ 这个函数接受3个参数:
            - script_name(str): 要执行的 Python 脚本的文件名。
                                ex: "train_0.py"
            - work_dir(str):    执行脚本的工作目录,默认为当前目录 "."。
                                ex: "./workspace/gpt-3.5-turbo-16k_False_0/detect-ai-generation"
            - device(str):      指定要使用的 GPU 设备,默认为 "0"。
    """
    
    print(f'\n{GREEN}----------------------- execute_script(script_name, work_dir = ".", device="0") 函数：开始 -----------------------{RESET}\n')
    # 想要执行的 ·python 脚本的地址
    script_path = os.path.join(workspace_task_dir, script_name)
    # print('workspace_task_dir =',  workspace_task_dir)   # ./workspace/gpt-3.5-turbo-16k_False_0/detect-ai-generation
    # print('script_name        =',  script_name)          # train_0.py
    print('script_path        = ', script_path)          # ./workspace/gpt-3.5-turbo-16k_False_0/detect-ai-generation/train_0.py
    print('os.path.exists(script_path) =', os.path.exists(script_path)) # True
    print('\n')
    if not os.path.exists(script_path):
        print(f'\n{GREEN}----------------------- execute_script(script_name, work_dir = ".", device="0") 函数：结束 -----------------------{RESET}\n')
        raise Exception(f"The file {script_name} does not exist.")
    try:
        # script_path = script_name
        device = device
        cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_name}"
        # cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_path}" # python: can't open file '/DS_Agent/DS-Agent/deployment/workspace/gpt-3.5-turbo-16k_False_0/detect-ai-generation/./workspace/gpt-3.5-turbo-16k_False_0/detect-ai-generation/train_0.py': [Errno 2] No such file or directory

        print(f'{BLUE}cmd = {cmd}{RESET}') # CUDA_VISIBLE_DEVICES=0 python -u prompt.py
        
        # (1) 使用 subprocess.Popen 来执行 cmd 命令
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=workspace_task_dir)


        # (2) 使用 selectors 模块来监视脚本的标准输出和标准错误流,以实时捕获和打印输出。它还会将输出存储在 stdout_lines 和 stderr_lines 列表中。
        stdout_lines = []
        stderr_lines = []

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        while process.poll() is None and selector.get_map():
            events = selector.select(timeout=1)

            for key, _ in events:
                line = key.fileobj.readline()
                if key.fileobj == process.stdout:
                    print("STDOUT:", line, end =" ")
                    stdout_lines.append(line)
                else:
                    print("STDERR:", line, end =" ")
                    stderr_lines.append(line)

        for line in process.stdout:
            line = line
            print("STDOUT:", line, end =" ")
            stdout_lines.append(line)
        for line in process.stderr:
            line = line
            print("STDERR:", line, end =" ")
            stderr_lines.append(line)

        # (3) 在脚本执行完毕后,函数会检查返回码 return_code。
            # (3).1 如果 return_code 不为 0 (i.e, -1),则认为执行失败,将标准错误流的输出作为观察结果返回。
            # (3).2 如果 return_code 为 0,将标准输出流的输出作为观察结果返回。
            # (3).3 如果 observation 为 0 & 为 0,则认为只有标准错误流有输出,将标准错误流的输出作为观察结果返回。
        return_code = process.returncode
        
        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            # printed to stderr only
            observation = "".join(stderr_lines)

        print('\n')
        print('observation =', observation) # final area under the ROC curve (AUROC) on validation set:  0.8563294523983053 Final AUROC on test set: 0.9902176523761718.
        print('return_code =', return_code) # 0
        print(f'\n{GREEN}----------------------- execute_script(script_name, work_dir = ".", device="0") 函数：结束 -----------------------{RESET}\n')
        return "The script has been executed. Here is the output:\n" + observation
        
    except Exception as e:
        print("++++", "Wrong!")
        print(f'\n{GREEN}----------------------- execute_script(script_name, work_dir = ".", device="0") 函数：结束 -----------------------{RESET}\n')
        raise Exception(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")
    
print(f'\n{YELLOW}==================================================================== execution.py: 结束 ==================================================================={RESET}\n')

#%%

# execute_script("prompt.py", workspace_task_dir = "./", device="0")

# 总结：execution.py 文件中定义了一个名为 execute_script 的函数,用于在指定的工作目录中执行一个 Python 脚本。
# execute_script 函数接受3个参数: script_name, workspace_task_dir, device
# 如果这个 .py 文件不存在,则会抛出异常 "The file {script_name} does not exist."
# 如果执行成功,则返回 "The script has been executed. Here is the output:\n" + observation
# 如果执行失败,则返回异常 "Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."


#%%
