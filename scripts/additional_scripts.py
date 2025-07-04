# import nltk
# nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/nltk_data')

# import json
# import os

# file_path = "/root/magic-pdf.json"

# new_config = {
#     "device-mode": "cuda",
#     "layout-config": {
#         "model": "layoutlmv3"
#     },
#     "formula-config": {
#         "mfd_model": "yolo_v8_mfd",
#         "mfr_model": "unimernet_small",
#         "enable": False
#     },
#     "table-config": {
#         "model": "tablemaster",
#         "enable": False,
#         "max_time": 400
#     }
# }

# if os.path.exists(file_path):
#     with open(file_path, "r", encoding="utf-8") as file:
#         try:
#             data = json.load(file) 
#         except json.JSONDecodeError:
#             data = {}
# else:
#     data = {}

# data.update(new_config)

# with open(file_path, "w", encoding="utf-8") as file:
#     json.dump(data, file, indent=4)

# print(f"File '{file_path}' has been updated.")