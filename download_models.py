#!/usr/bin/env python3
"""
手动下载模型脚本
用于避免在运行时遇到 HTTP 429 错误
"""

import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download
import requests

def setup_cache_dirs():
    """设置缓存目录"""
    cache_dirs = [
        './models/transformers_cache',
        './models/huggingface_cache', 
        './models/huggingface_hub_cache'
    ]
    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"创建缓存目录: {cache_dir}")

def download_with_retry(model_name, cache_dir, max_retries=5):
    """带重试的模型下载"""
    for attempt in range(max_retries):
        try:
            print(f"正在下载模型 {model_name} (尝试 {attempt + 1}/{max_retries})...")
            
            # 使用 snapshot_download 下载整个模型
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_dir=os.path.join(cache_dir, model_name.replace('/', '--')),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"模型 {model_name} 下载成功!")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** attempt  # 指数退避
                print(f"遇到 HTTP 429 错误，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"HTTP 错误: {e}")
                return False
        except Exception as e:
            print(f"下载失败: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"模型 {model_name} 下载失败，已达到最大重试次数")
                return False
    
    return False

def main():
    """主函数"""
    print("开始下载模型...")
    
    # 设置环境变量
    os.environ['HF_HUB_CACHE'] = './models/huggingface_hub_cache'
    os.environ['TRANSFORMERS_CACHE'] = './models/transformers_cache'
    
    # 创建缓存目录
    setup_cache_dirs()
    
    # 要下载的模型列表
    models = [
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    success_count = 0
    for model in models:
        if download_with_retry(model, './models/huggingface_hub_cache'):
            success_count += 1
    
    print(f"\n下载完成! 成功下载 {success_count}/{len(models)} 个模型")
    
    if success_count == len(models):
        print("所有模型下载成功! 现在可以运行应用程序了。")
    else:
        print("部分模型下载失败，请检查网络连接或稍后重试。")

if __name__ == "__main__":
    main() 