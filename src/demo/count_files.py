import os

def count_files_in_folders(parent_folder):
    folder_counts = {}

    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            folder_counts[folder] = num_files

    return folder_counts

# 指定arxiv_downloads文件夹路径
parent_folder = "arxiv_downloads"
counts = count_files_in_folders(parent_folder)

# 输出统计结果
for folder, num_files in counts.items():
    print(f"{folder}: {num_files} files")
