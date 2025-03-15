# 使用官方的轻量镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 复制项目依赖文件
COPY requirements.txt /app/

# 安装 Python 项目依赖及其他组件
RUN pip install --no-cache-dir --no-deps -r requirements.txt && \
    pip install --no-cache-dir bertopic && \
    pip install --no-cache-dir -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com && \
    pip install --no-cache-dir unstructured==0.16.10 && \
    pip install --no-cache-dir huggingface_hub && \
    pip uninstall -y pdfminer && \
    pip uninstall -y pdfminer-six && \
    pip install pdfminer-six && \
    pip uninstall -y opencv-python-headless && \
    pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    python -m nltk.downloader punkt

# 下载模型文件并调整存储路径
RUN wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py && \
    python download_models_hf.py && \
    rm -rf /root/.cache/pip && \ 
    python scripts/additional_scripts.py

# 复制项目代码
COPY . /app/

# 暴露服务端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s CMD curl -f http://localhost:8001/ || exit 1

# 启动命令
CMD ["python", "src/manage.py", "runserver", "0.0.0.0:8001"]