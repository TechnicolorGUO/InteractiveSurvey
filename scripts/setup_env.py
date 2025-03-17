#!/usr/bin/env python3
import subprocess
import sys

# 按顺序构造 pip 安装命令列表
commands = [
    # 安装 unstructured==0.16.10
    [sys.executable, "-m", "pip", "install", "unstructured==0.16.10"],  
    # 安装 requests==2.32.3
    [sys.executable, "-m", "pip", "install", "requests==2.32.3"],
    # 安装 chromadb==0.5.4
    [sys.executable, "-m", "pip", "install", "chromadb==0.5.4"],
    # 安装 langchain-huggingface==0.1.2（注意去除中间的空格）
    [sys.executable, "-m", "pip", "install", "langchain-huggingface==0.1.2"],
    # 安装 markdown_pdf==1.3
    [sys.executable, "-m", "pip", "install", "markdown_pdf==1.3"],
    # 安装 bertopic==0.16.3
    [sys.executable, "-m", "pip", "install", "bertopic==0.16.3"],
    # 强制升级安装 langchain-community
    [sys.executable, "-m", "pip", "install", "-U", "langchain-community"],
    # 使用 pytorch 官方源（cu118 版本）强制重装 torch、torchvision 和 numpy (<2.0.0)
    [sys.executable, "-m", "pip", "install", "--force-reinstall", "torch==2.3.1", "torchvision==0.18.1", "numpy<2.0.0", "--index-url", "https://download.pytorch.org/whl/cu118"],
    # 再次使用 wheels.myhloli.com 额外源升级安装 magic-pdf[full]
    [sys.executable, "-m", "pip", "install", "-U", "magic-pdf[full]", "--extra-index-url", "https://wheels.myhloli.com"],
    # 安装 paddlepaddle-gpu==2.6.1
    [sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==2.6.1"],
    # 安装 Django==2.2.5
    [sys.executable, "-m", "pip", "install", "Django==2.2.5"]
]

def run_commands(cmds):
    for cmd in cmds:
        cmd_str = " ".join(cmd)
        print(f"正在执行命令: {cmd_str}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"命令执行失败: {cmd_str}")
            sys.exit(1)

if __name__ == "__main__":
    run_commands(commands)