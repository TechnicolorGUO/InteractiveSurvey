#!/usr/bin/env python3
import subprocess
import sys
import shutil
import os

def clear_pip_cache():
    print("🧹 Cleaning...")
    try:
        # 获取 pip 缓存目录
        result = subprocess.run([sys.executable, "-m", "pip", "cache", "dir"], stdout=subprocess.PIPE, check=True, text=True)
        cache_dir = result.stdout.strip()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"✅ : {cache_dir}")
        else:
            print("No cache dir found.")
    except Exception as e:
        print(f"❌ {e}")

# 按顺序构造 pip 安装命令列表（全部加上 --no-cache-dir）
commands = [
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "unstructured==0.16.10"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "requests==2.32.3"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "chromadb==0.5.4"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "langchain-huggingface==0.1.2"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "markdown_pdf==1.3"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "bertopic==0.16.3"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-U", "langchain-community"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "torch==2.3.1", "torchvision==0.18.1", "numpy<2.0.0", "--index-url", "https://download.pytorch.org/whl/cu118"],
    # 安装 MinerU 2.0 (替代旧的 magic-pdf)
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-U", "mineru[core]"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "Django==2.2.5"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "graphviz"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "einops"],
    [sys.executable, "-m", "pip", "install", "--no-cache-dir", "markdown"],
]

def run_commands(cmds):
    for cmd in cmds:
        cmd_str = " ".join(cmd)
        print(f"🚀 {cmd_str}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ {cmd_str}")
            sys.exit(1)

if __name__ == "__main__":
    clear_pip_cache()
    run_commands(commands)
