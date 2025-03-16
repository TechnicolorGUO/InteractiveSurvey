import json
import os
import re
from urllib.parse import quote

def insert_ref_images(json_path, ref_names, text):
    """
    参数:
      json_path: JSON 文件路径，文件内容格式例如：
                 {
                   "Accelerating federated learning with data and model parallelism in edge computing":
                     "src/static/data/md/test/Accelerating federated learning with data and model parallelism in edge computing/auto/images/xxx.jpg",
                   ... 
                 }
      ref_names: 引用名称列表，其中第 1 个元素对应 [1]，第 2 个对应 [2]，以此类推。
      text: 包含类似 [1]、[2] 等引用的 Markdown 文本。

    返回:
      修改后的文本字符串。在每个引用标记首次出现行的下方插入对应的 HTML 代码块，
      格式如下：
      
      <div style="text-align:center">
          <img src="relative_path" alt="the flow chart of [ref_name]" style="width:50%;"/>
      </div>
      <div style="text-align:center">
          Fig [ref_num]: The flow chart of [ref_name]
      </div>
      
      其中 [ref_num] 为引用编号（ref_names 中的 1-based index），[ref_name] 为引用名称。
      
    说明：
      1. 代码中自动计算项目根目录，假设项目根目录为 views 文件所在目录的上两级目录。
      2. 图片的最终相对路径是相对于运行该脚本（即 src/demo/views.py）的路径。
    """

    # 假设项目根目录是 views 文件所在目录的上两级目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    # 加载 JSON 文件内容
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            img_mapping = json.load(f)
    except Exception as e:
        raise Exception(f"加载 JSON 文件出错: {e}")

    inserted_refs = {}  # 记录每个引用标记是否已插入图片
    lines = text.splitlines()
    new_lines = []
    # 匹配类似 [1]、[2] 等引用标记
    ref_pattern = re.compile(r'\[(\d+)\]')

    for line in lines:
        new_lines.append(line)
        matches = ref_pattern.findall(line)
        for ref_num_str in matches:
            try:
                ref_num = int(ref_num_str)
            except ValueError:
                continue

            # 仅第一次出现时插入 HTML 图片块和 caption
            if ref_num not in inserted_refs:
                inserted_refs[ref_num] = True

                if 1 <= ref_num <= len(ref_names):
                    ref_name = ref_names[ref_num - 1]
                    jpg_path = img_mapping.get(ref_name, "")
                else:
                    ref_name = f"ref_{ref_num}"
                    jpg_path = ""
                
                if jpg_path:
                    # 计算图片的绝对路径。注意：这里使用项目根目录拼接 JSON 中的图片路径
                    absolute_jpg_path = os.path.join(project_root, jpg_path)
                    # 计算图片相对于 views.py 所在目录（即 base_dir）的相对路径
                    relative_path = os.path.relpath(absolute_jpg_path, start=base_dir)
                    # URL编码空格等特殊字符（保留斜杠）
                    relative_path_url = quote(relative_path, safe="/")

                    html_block = (
                        f"<div style=\"text-align:center\">\n"
                        f"    <img src=\"{relative_path_url}\" alt=\"the flow chart of {ref_name}\" style=\"width:50%;\"/>\n"
                        f"</div>\n"
                        f"<div style=\"text-align:center\">\n"
                        f"    Fig {ref_num}: The flow chart of {ref_name}\n"
                        f"</div>"
                    )
                    new_lines.append(html_block)
                    new_lines.append("")  # 增加空行分隔

    return "\n".join(new_lines)


# 示例用法
if __name__ == "__main__":
    # Markdown 文件路径
    md_file_path = "src/static/data/info/test/survey_test_processed.md"
    # JSON 文件路径
    json_file_path = "src/static/data/info/test/flowchart_results.json"

    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"错误: Markdown 文件 {md_file_path} 未找到！")
        text = ""

    ref_names = [
        "An explainable federated learning and blockchain based secure credit modeling method",
        "Bafl a blockchain based asynchronous",
        "Biscotti a blockchain system for private and secure federated learning",
        "Blockdfl a blockchain based fully decentralized peer to peer",
        "Accelerating blockchain enabled federated learning with clustered clients",
        "A fast blockchain based federated learning framework with compressed communications"
    ]

    result = insert_ref_images(json_file_path, ref_names, text)
    print("修改后的文本为：\n")
    print(result)