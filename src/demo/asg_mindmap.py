import json
import re

def generate_mermaid_html(json_path, output_html_path):
    """
    该函数读取指定的 JSON 文件（json_path），解析并构造大纲的树状结构，
    之后转换为 Mermaid mindmap 格式，并最终生成包含 Mermaid 代码的 HTML 文件（output_html_path）。
    """
    # 1. 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. 获取 outline 字符串（内容类似 "[1, '1 Abstract'], [1, '2 Introduction'], [1, '3 ...']"）
    outline_str = data.get("outline", "")
    
    # 3. 利用正则表达式解析每一项，匹配格式：[层级, '标题']
    pattern = re.compile(r"\[(\d+),\s*'([^']+)'\]")
    items = pattern.findall(outline_str)
    # 转换为列表，每个元素为 (int(层级), 标题)
    items = [(int(level), title) for level, title in items]
    
    # 4. 过滤掉不需要的条目：
    undesired_titles = {"1 Abstract", "2 Introduction", "6 Future Directions", "7 Conclusion"}
    filtered_items = [(lvl, title) for lvl, title in items if not (lvl == 1 and title in undesired_titles)]
    
    # 5. 利用栈构建树状结构，每个节点为字典 {"title": 标题, "children": []}
    tree = []
    stack = []  # 存储 (层级, 节点)
    for lvl, title in filtered_items:
        node = {"title": title, "children": []}
        # 弹出不属于当前节点父层级的节点
        while stack and lvl <= stack[-1][0]:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            tree.append(node)
        stack.append((lvl, node))
    
    # 6. 递归将树状结构转换为 Mermaid mindmap 格式的文本
    # 定义一个根节点，这里作为 mindmap 的中心节点
    mermaid_lines = ["mindmap", "  root((Document Outline))"]

    def tree_to_mermaid(node, indent_level):
        indent = "  " * indent_level
        # 生成当前节点行，注意这里不添加前缀符号，直接利用缩进作为层级标识
        line = f"{indent}{node['title']}"
        lines = [line]
        for child in node.get("children", []):
            child_lines = tree_to_mermaid(child, indent_level + 1)
            lines.extend(child_lines)
        return lines

    # 根据每个顶级节点生成 Mermaid 文本，顶级节点在 root 下（设定 indent_level 为2）
    for top_node in tree:
        mermaid_lines.extend(tree_to_mermaid(top_node, 2))
    
    mermaid_text = "\n".join(mermaid_lines)
    
    # 7. 构造完整 HTML 文件，内含 Mermaid 配置和代码块
    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mermaid Mindmap</title>
  <!-- 从 CDN 加载 Mermaid，注意：mindmap 是 Mermaid 的实验性功能，请确保使用新版 Mermaid -->
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true }});
  </script>
  <style>
      body {{
          font-family: Arial, sans-serif;
          background: #fafafa;
          padding: 20px;
      }}
      pre {{
          background: #f0f0f0;
          padding: 10px;
          border-radius: 5px;
      }}
  </style>
</head>
<body>
  <!-- Mermaid 流程图代码块 -->
  <div class="mermaid">
{mermaid_text}
  </div>
</body>
</html>
"""
    # 8. 写入 HTML 文件
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print("生成 HTML 文件：", output_html_path)

# 使用示例：
# 假设 JSON 文件为 "outline.json"，生成的 HTML 文件为 "mermaid_mindmap.html"
# 调用以下函数即可生成 HTML 文件：
# generate_mermaid_html("src/static/data/txt/test_1/outline.json", "mermaid_mindmap.html")
import re

def insert_html_before_first_intro(html_path: str, md_string: str) -> str:
    """
    读取 HTML 文件内容，并在 Markdown 的第一个 '2 Introduction' 之前插入 HTML 内容。

    :param html_path: HTML 文件路径
    :param md_string: Markdown 字符串
    :return: 修改后的 Markdown 字符串
    """
    # 读取 HTML 文件内容
    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read().strip()

    # 查找第一个 "2 Introduction"
    match = re.search(r"(^|\n)2 Introduction", md_string)

    # 如果找不到 "2 Introduction"，返回原始 Markdown
    if not match:
        return md_string

    first_intro_index = match.start()  # 获取第一个匹配的起始索引

    # 在该索引之前插入 HTML 内容
    updated_md = md_string[:first_intro_index] + f"\n\n{html_content}\n\n" + md_string[first_intro_index:]

    return updated_md

# 示例 Markdown 内容
markdown_text = """# My Document

Some introduction text.

2 Introduction

This is the first '2 Introduction' section.

Some more text.

2 Introduction

This is the second '2 Introduction' section.
"""

# 假设 HTML 文件路径
html_file_path = "mermaid_mindmap.html"

# 运行函数
updated_markdown = insert_html_before_first_intro(html_file_path, markdown_text)

# 打印结果
print(updated_markdown)