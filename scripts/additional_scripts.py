import nltk
nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/nltk_data')

import json
import os

# 文件路径
file_path = "/root/magic-pdf.json"

# 要添加或覆盖的内容
new_config = {
    "layout-config": {
        "model": "layoutlmv3"
    },
    "formula-config": {
        "mfd_model": "yolo_v8_mfd",
        "mfr_model": "unimernet_small",
        "enable": False
    },
    "table-config": {
        "model": "tablemaster",
        "enable": False,
        "max_time": 400
    }
}

# 检查文件是否存在
if os.path.exists(file_path):
    # 如果文件存在，读取内容
    with open(file_path, "r") as file:
        try:
            data = json.load(file)  # 读取 JSON 文件内容
        except json.JSONDecodeError:
            # 如果文件不是有效的 JSON，初始化为空字典
            data = {}
else:
    # 如果文件不存在，初始化为空字典
    data = {}

# 更新或添加键值
data.update(new_config)

# 将更新后的内容写回文件
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)

print(f"File '{file_path}' has been updated.")