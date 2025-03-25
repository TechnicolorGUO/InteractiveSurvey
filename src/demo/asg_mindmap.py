import json
import re
import textwrap
from graphviz import Digraph
import os

def wrap_text(text, max_chars):
    """
    对文本进行自动换行包装，每行最大字符数为 max_chars。
    """
    return textwrap.fill(text, width=max_chars)

def parse_md_refs(md_content):
    """
    解析 Markdown 内容，提取以 x.y.z 格式标题对应的引用。
    
    对于每个满足格式的 section，其内容中所有形如 [数字] 的引用
    将被抽取出来，去重后按数字升序排序，生成类似 "[1,2,3]" 的引用字符串。
    
    如果遇到 undesired header（如 "6 Future Directions" 或 "7 Conclusion"），
    则停止后续内容的解析，确保最后一个 section 仅包含到该 header 之前的内容。

    返回字典，键为 section 编号（例如 "3.1.1"），值为引用字符串（例如 "[1,2,3]"）。
    """
    ref_dict = {}

    # 处理 Markdown 内容（按行拆分）
    lines = md_content.split("\n") if md_content else []

    # 匹配 Markdown 标题中以 x.y.z 开头的叶节点（例如 "5.1.1 Neural Topic..."）
    section_header_regex = re.compile(r'^\s*#+\s*(\d+\.\d+\.\d+).*')
    # 匹配 undesired header，如 "6 Future Directions" 或 "7 Conclusion"
    undesired_header_regex = re.compile(r'^\s*#+\s*(6 Future Directions|7 Conclusion)\b')
    # 匹配引用，例如 [数字]
    ref_pattern = re.compile(r'\[(\d+)\]')
    
    current_section = None
    current_content = []

    for line in lines:
        # 如果检测到 undesired header，则先处理当前 section，再退出循环
        if undesired_header_regex.match(line):
            break

        header_match = section_header_regex.match(line)
        if header_match:
            # 处理上一个 section
            if current_section is not None:
                all_refs = [int(num) for content_line in current_content for num in ref_pattern.findall(content_line)]
                if all_refs:
                    ref_dict[current_section] = "[" + ",".join(map(str, sorted(set(all_refs)))) + "]"
            
            # 更新当前 section
            current_section = header_match.group(1)
            current_content = []
        else:
            if current_section is not None:
                current_content.append(line)

    # 处理最后一个 section
    if current_section is not None and current_content:
        all_refs = [int(num) for content_line in current_content for num in ref_pattern.findall(content_line)]
        if all_refs:
            ref_dict[current_section] = "[" + ",".join(map(str, sorted(set(all_refs)))) + "]"

    return ref_dict

def generate_graphviz_png(json_path, output_png_path, md_content=None, title="Document Outline", max_root_chars=20):
    """
    从 JSON 文件中读取大纲，构造树状结构，并生成 mindmap 的 PNG 图片。

    如果提供了 md_content，则根据 Markdown 内容中以 x.y.z 格式标题对应的引用，
    在生成 mindmap 时，对于叶节点（没有子节点且标题以 x.y.z 开头）的标签，
    在原文本后追加一个换行，然后添加引用信息（例如 "[1,2,3]"），
    且引用经过数字排序。
    
    同时，仅对根节点文本进行自动换行包装，以限制根节点的最大宽度，
    其它节点保持原始文本格式。

    参数:
      json_path: JSON 文件路径（包含大纲）
      output_png_path: 输出 PNG 文件路径（不带后缀）
      md_content: Markdown 文本内容（字符串），可选
      title: 用于替换 mindmap 中根节点的标题，默认 "Document Outline"
      max_root_chars: 限制根节点每行最大字符数，默认 20
    """
    # 解析 Markdown 内容的引用
    ref_dict = parse_md_refs(md_content) if md_content else {}

    # 读取 JSON 大纲
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    outline_str = data.get("outline", "")

    # 解析形如 [层级, '标题'] 的项
    pattern = re.compile(r"\[(\d+),\s*'([^']+)'\]")
    items = pattern.findall(outline_str)
    items = [(int(level), title) for level, title in items]

    # 不需要的标题关键词
    undesired_keywords = {"Abstract", "Introduction", "Future Directions", "Conclusion"}
    # 过滤掉不需要的条目
    filtered_items = [
        (lvl, title) for lvl, title in items
        if not re.match(r"^\d+\s+(.+)", title) or re.match(r"^\d+\s+(.+)", title).group(1) not in undesired_keywords
    ]

    # 构造树状结构
    tree = []
    stack = []
    for lvl, title_item in filtered_items:
        node = {"title": title_item, "children": []}
        while stack and lvl <= stack[-1][0]:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            tree.append(node)
        stack.append((lvl, node))

    # 生成 Mindmap
    dot = Digraph(comment=title, format='png', engine='dot')
    dot.graph_attr.update(rankdir='LR', splines='ortho', bgcolor='white', dpi="150")
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', color='gray')
    dot.edge_attr.update(arrowhead='none', color="black")
    
    # 处理根节点
    wrapped_title = wrap_text(title, max_root_chars)
    dot.node('root', label=wrapped_title, shape='ellipse', style='filled', fillcolor='lightgray')

    node_counter = [0]
    section_pattern = re.compile(r'^(\d+\.\d+\.\d+)')

    def add_nodes(node, parent_id):
        current_id = f'node_{node_counter[0]}'
        node_counter[0] += 1
        safe_label = node['title'].replace('"', r'\"')
        
        # 如果是叶节点且标题以 x.y.z 开头，则追加引用信息（如果存在）
        if not node["children"]:
            m = section_pattern.match(safe_label)
            if m:
                section_id = m.group(1)
                if section_id in ref_dict:
                    safe_label += "\n" + ref_dict[section_id]
        
        dot.node(current_id, label=safe_label)
        dot.edge(parent_id, current_id)
        for child in node.get("children", []):
            add_nodes(child, current_id)

    for top_node in tree:
        add_nodes(top_node, "root")

    dot.render(output_png_path, cleanup=True)
    print("生成 PNG 文件：", output_png_path + ".png")
    return output_png_path + ".png"



def insert_outline_image(png_path, md_content, survey_title):
    """
    在给定的 Markdown 内容字符串中查找 "2 Introduction" 这一行，
    然后在该位置之前插入 outline 图片的 HTML 代码块，确保渲染时
    HTML 块与后续 Markdown 内容间有足够空行分隔开。

    参数：
      png_path: 要插入的 PNG 图片路径，将作为 img 的 src 属性值。
      md_content: Markdown 文件内容字符串。
      survey_title: 用于生成图片说明文字的问卷标题。

    插入的 HTML 格式如下：

      <div style="text-align:center">
          <img src="{png_path}" alt="Outline" style="width:100%;"/>
      </div>
      <div style="text-align:center">
          Fig 1. The outline of the {survey_title}
      </div>

    函数返回更新后的 Markdown 内容字符串。
    """

    # 将 Markdown 内容字符串分割成行（保留换行符）
    lines = md_content.splitlines(keepends=True)
    print(lines)

    # 查找包含 "2 Introduction" 的行的索引
    intro_index = None
    for i, line in enumerate(lines):
        if '2 Introduction' in line:
            intro_index = i
            break

    if intro_index is None:
        print("没有找到 '2 Introduction' 这一行！")
        return md_content

    # 确保路径中的反斜杠被替换成正斜杠
    png_path_fixed = png_path.replace("\\", "/")
    
    # 构造需要插入的 HTML 代码块，在前后增加空行
    html_snippet = (
        "\n\n"  # 添加换行确保与上文/下文分隔
        f'<div style="text-align:center">\n'
        f'    <img src="{png_path_fixed}" alt="Outline" style="width:100%;"/>\n'
        f'</div>\n'
        f'<div style="text-align:center">\n'
        f'    Fig 1. The outline of the {survey_title}\n'
        f'</div>\n'
        "\n"  # 再添加一个空行确保与下方内容分隔
    )
    
    print(f"将在第 {intro_index} 行插入如下 HTML 代码块（插入在 '2 Introduction' 之前）：\n{html_snippet}")
    
    # 在找到的 "2 Introduction" 这一行之前插入 html_snippet
    lines.insert(intro_index, html_snippet)

    # 合并所有行，构造更新后的 Markdown 内容
    updated_md = "".join(lines)
    
    print("已在 Markdown 内容中插入 outline 图片。")
    return updated_md

def insert_outline_figure(png_path, tex_content, survey_title):
    """
    在给定的 TeX 文件内容字符串中查找 "2 Introduction" 这一行，
    然后在其之前插入一个跨页(双栏)的 figure* 环境，包括整页显示的图片。
    它将生成类似如下 LaTeX 片段：
    
    \begin{figure*}[htbp]
      \centering
      \includegraphics[width=\textwidth]{path/to/xxx.png}
      \caption{Fig 1. The outline of the XXX}
    \end{figure*}

    参数：
      png_path: 要插入的 PNG 图片路径
      tex_content: TeX 文件内容字符串
      survey_title: 用于生成图片 caption 的文献/问卷标题

    返回:
      更新后的 TeX 文本字符串
    """

    # 将 TeX 内容逐行分割（保留换行符）
    lines = tex_content.splitlines(keepends=True)

    # 查找包含 "2 Introduction" 的行索引
    intro_index = None
    for i, line in enumerate(lines):
        if 'Introduction' in line:
            intro_index = i
            break

    # 如果找不到，就直接返回原文
    if intro_index is None:
        print("没有找到 'Introduction' 这一行，未执行插入。")
        return tex_content

    # 构造 TeX 的 figure* 代码块
    # 为确保整页，可用 [p] 或者 [htbp]，具体可根据排版需要调整
    # 也可替换成普通 \begin{figure} ... \end{figure}，如果不需要跨双栏
    figure_block = (
        "\n"  # 加一个空行，确保与上文分隔
        "\\begin{figure*}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{png_path}}}\n"
        f"  \\caption{{The outline of our survey: {survey_title}}}\n"
        "\\end{figure*}\n\n"  # 再留一个空行分隔
    )

    # 在找到的 "2 Introduction" 所在行之前插入 figure 环境
    lines.insert(intro_index, figure_block)

    # 重新拼接所有行
    updated_tex = "".join(lines)
    return updated_tex
# 使用示例：
# if __name__ == "__main__":
#     png_path = 'src/static/data/info/test_4/outline.png'
#     md_content = ''
#     survey_title = "My Survey Title"
#     updated_md = insert_outline_image(png_path, md_content, survey_title)
# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    json_path = os.path.join("src", "static", "data", "txt", 'test_2', "outline.json")
    output_png_path = os.path.join("src", "static", "data", "info", 'test_2', "outline")
    md_path = os.path.join("src", "static", "data", "info", 'test_2', f"survey_{'test_2'}_processed.md")
    flowchart_results_path = os.path.join("src", "static", "data", "info", 'test_2', "flowchart_results.json")
    png_path = generate_graphviz_png(
            json_path=json_path,
            output_png_path=output_png_path,
            md_path=md_path,
            title='test',
            max_root_chars=30
        )

    # generate_graphviz_png(json_file_path, output_png_file, md_file_path, title=mindmap_title, max_root_chars=20)