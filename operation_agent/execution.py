import os
import subprocess
import selectors
import sys


def execute_multifile_package(files_dict: dict, main_file: str, base_dir: str, device: str = "0"):
    """保存多文件包到 base_dir，然后执行 main_file。

    所有文件路径自动压平到 base_dir 根目录（去除子目录前缀），
    避免 LLM 生成 src/data/xxx.py 这类嵌套路径导致找不到文件。
    """
    os.makedirs(base_dir, exist_ok=True)

    # Flatten: strip any leading directory components (e.g. src/data/foo.py → foo.py)
    flat_files = {}
    for fname, code in files_dict.items():
        flat_name = os.path.basename(fname)
        flat_files[flat_name] = code

    for flat_name, code in flat_files.items():
        fpath = os.path.join(base_dir, flat_name)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(code)

    # Resolve main_file to its flat basename
    flat_main = os.path.basename(main_file)
    if flat_main not in flat_files:
        return -1, f"Main file '{flat_main}' not found in generated files: {list(flat_files.keys())}"

    # Execute from the repository root so generated code can use paths like
    # agent_workspace/datasets/... consistently. Python still adds the script
    # directory to sys.path, so flat sibling imports (from config import ...)
    # continue to work.
    return execute_script(os.path.abspath(os.path.join(base_dir, flat_main)), work_dir=".", device=device)


def execute_script(script_name, work_dir = ".", device="0"):    
    if not os.path.exists(os.path.join(work_dir, script_name)):
        raise Exception(f"The file {script_name} does not exist.")
    try:
        script_path = script_name
        device = device        
        python_executable = sys.executable
        pythonpath = os.path.abspath(work_dir)
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath = f"{pythonpath}{os.pathsep}{existing_pythonpath}"
        cmd = f"PYTHONPATH={pythonpath} CUDA_VISIBLE_DEVICES={device} {python_executable} -u {script_path}"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=work_dir)

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
                    # print("STDOUT:", line, end =" ")
                    stdout_lines.append(line)
                else:
                    # print("STDERR:", line, end =" ")
                    stderr_lines.append(line)

        for line in process.stdout:
            line = line
            # print("STDOUT:", line, end =" ")
            stdout_lines.append(line)
        for line in process.stderr:
            line = line
            # print("STDERR:", line, end =" ")
            stderr_lines.append(line)

        return_code = process.returncode

        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            # printed to stderr only
            observation = "".join(stderr_lines)
        return return_code, "The script has been executed. Here is the output:\n" + observation
    
    except Exception as e:
        print("++++", "Wrong!")
        # raise Exception(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")
        return -1, f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."
