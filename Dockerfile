# 使用官方的轻量镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 更新系统并安装依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential \
    graphviz texlive-xetex texlive-fonts-recommended texlive-latex-recommended libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# 复制项目依赖文件并安装环境
COPY scripts/ /app/
RUN python setup_env.py

# 下载模型文件并调整存储路径
# RUN wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py && \
#     python download_models_hf.py && \
#     rm -rf /root/.cache/pip && \
#     python additional_scripts.py

# 复制项目代码
COPY . /app/

# 暴露服务端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s CMD curl -f http://localhost:8001/ || exit 1

# 启动命令
CMD ["python", "src/manage.py", "runserver", "0.0.0.0:8001"]