#!/usr/bin/env python3
import subprocess
import sys

def uninstall_old_packages():
    """卸载旧版本的 magic-pdf 相关包"""
    
    old_packages = [
        "magic-pdf",
        "magic-pdf[full]",
        "paddlepaddle",
        "paddleocr", 
        "layoutparser",
    ]
    
    print("🗑️ 卸载旧版本的 MinerU 相关包...")
    
    for package in old_packages:
        try:
            print(f"🔄 尝试卸载 {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", package], 
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"✅ 成功卸载 {package}")
            else:
                print(f"ℹ️ {package} 未安装或已卸载")
        except Exception as e:
            print(f"⚠️ 卸载 {package} 时出错: {e}")
    
    # 清理 pip 缓存
    try:
        print("🧹 清理 pip 缓存...")
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=True)
        print("✅ 缓存清理完成")
    except Exception as e:
        print(f"⚠️ 清理缓存时出错: {e}")

if __name__ == "__main__":
    uninstall_old_packages()
    print("🎉 卸载完成！现在可以安装新版本的 MinerU 了。") 