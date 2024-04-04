import os
import subprocess
import selectors

# 这段代码定义了一个名为 execute_script 的函数,用于在指定的工作目录中执行一个 Python 脚本。
def execute_script(script_name, work_dir = ".", device="0"):
    """ 这个函数接受3个参数:
            - script_name(str): 要执行的 Python 脚本的文件名。
            - work_dir(str):    执行脚本的工作目录,默认为当前目录 "."。
            - device(str):      指定要使用的 GPU 设备,默认为 "0"。
    """
    # 想要执行的 ·python 脚本的地址
    script_path = os.path.join(work_dir, script_name)
    print('script_path = ', script_path) # ./train_{idx}.py
    if not os.path.exists(script_path):
        raise Exception(f"The file {script_name} does not exist.")
    try:
        # script_path = script_name
        device = device
        cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_name}"
        # cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_path}"
        
        # 使用 subprocess.Popen 来执行 cmd 命令
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=work_dir)

        stdout_lines = []
        stderr_lines = []

        # 该函数使用 selectors 模块来监视脚本的标准输出和标准错误流,以实时捕获和打印输出。它还会将输出存储在 stdout_lines 和 stderr_lines 列表中。
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

        # 在脚本执行完毕后,函数会检查返回码 return_code。如果返回码不为 0,则认为执行失败,将标准错误流的输出作为观察结果返回。
        否则,将标准输出流的输出作为观察结果返回。
        如果两者都为空,而且返回码为 0,则认为只有标准错误流有输出,将标准错误流的输出作为观察结果返回。
        return_code = process.returncode

        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            # printed to stderr only
            observation = "".join(stderr_lines)
        return "The script has been executed. Here is the output:\n" + observation
    except Exception as e:
        print("++++", "Wrong!")
        raise Exception(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")
