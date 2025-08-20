import re
import subprocess
import os

from openai import OpenAI
import dotenv
from .asg_add_flowchart import insert_tex_images
from .asg_mindmap import insert_outline_figure


def _remove_div_blocks(lines):
    """
    从给定的行列表中，移除所有形如:
      <div style="...">
          ... (若干行)
      </div>
    的 HTML 块（含首尾 <div> ... </div>）整段跳过。
    返回处理后的新行列表。
    """
    new_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # 如果该行以 <div style= 开头，则进入跳过模式
        if line.strip().startswith("<div style="):
            # 跳过本行
            i += 1
            # 一直向后找，直到遇到 '</div>' 行
            while i < n and not lines[i].strip().startswith("</div>"):
                i += 1
            # 这里再跳过 '</div>' 那一行
            i += 1
        else:
            new_lines.append(line)
            i += 1

    return new_lines

def _convert_setext_to_atx(lines):
    """
    将形如:

        标题文字
        ===
    
    转换为:
    
        # 标题文字

    将形如:

        标题文字
        ---

    转换为:

        ## 标题文字
    """
    setext_equal_pattern = re.compile(r'^\s*=+\s*$')  # 匹配全 `===`
    setext_dash_pattern  = re.compile(r'^\s*-+\s*$')  # 匹配全 `---`

    new_lines = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if i < n - 1:
            next_line = lines[i + 1].strip()
            # 若下一行是 ===
            if setext_equal_pattern.match(next_line):
                heading_text = line.strip()
                new_lines.append(f"# {heading_text}")
                i += 2  # 跳过下一行
                continue
            # 若下一行是 ---
            if setext_dash_pattern.match(next_line):
                heading_text = line.strip()
                new_lines.append(f"## {heading_text}")
                i += 2
                continue
        # 否则不改动
        new_lines.append(line)
        i += 1

    return new_lines

def preprocess_md(md_input_path: str, md_output_path: str = None) -> str:
    """
    预处理一个 Markdown 文件：
      1. 移除所有 <div style="..."> ... </div> 这类 HTML 块
      2. 将 setext 标题 (===, ---) 转为 ATX 标题 (#, ##)
      3. 覆盖写回或输出到新文件
    
    参数:
      md_input_path: 原始 Markdown 文件路径
      md_output_path: 处理后要写出的 Markdown 文件路径; 若为 None 则覆盖原始文件.

    返回:
      str: 返回处理后 Markdown 文件的实际写出路径 (md_output_path).
    """
    if md_output_path is None:
        md_output_path = md_input_path
    
    # 1) 读入行
    with open(md_input_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    # 2) 移除 <div style="..."> ... </div> 片段
    lines_no_div = _remove_div_blocks(lines)

    # 3) 将 setext 标题转换为 ATX
    lines_atx = _convert_setext_to_atx(lines_no_div)

    # 4) 写出
    with open(md_output_path, 'w', encoding='utf-8') as f:
        for ln in lines_atx:
            f.write(ln + "\n")

    return md_output_path

def search_sections(md_path: str):
    """
    解析仅含 ATX 风格标题的 Markdown 文件，返回一个列表，
    每个元素是一个三元组: (level, heading_text, content_string)
    
    说明:
      - 标题行形如 "# 标题"、"## 标题"、"### 标题" 等（在井号后有一个空格）。
      - level = (井号个数 - 1)，即 "# -> level=0"、"## -> level=1"、"### -> level=2" ...
      - 移除类似 "3.1.3 "、"2.10.1 " 这类数字点前缀（含其后空格）。
      - content_string 为该标题之后、直到下一个标题行或文件结束为止的所有文本（换行拼接）。
    """

    # 用于匹配 ATX 标题（如 "# 标题", "## 3.1.3 标题" 等）
    atx_pattern = re.compile(r'^(#+)\s+(.*)$')
    
    # 用于去除标题前缀的数字.数字.数字... (可能有空格)
    # 示例匹配: "3.1.3 "、"2.10.1 " 等
    leading_numbers_pattern = re.compile(r'^\d+(\.\d+)*\s*')

    # 读入行
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    sections = []
    i = 0
    n = len(lines)

    def gather_content(start_idx: int):
        """
        从 start_idx 开始，收集正文，直到遇到下一个 ATX 标题或文档末尾。
        返回 (content_string, end_idx).
        """
        content_lines = []
        idx = start_idx
        while idx < n:
            line = lines[idx].rstrip()
            # 如果此行匹配到 ATX 标题模式，则停止收集正文
            if atx_pattern.match(line):
                break
            content_lines.append(lines[idx])
            idx += 1
        return "\n".join(content_lines), idx

    while i < n:
        line = lines[i].rstrip()

        # 判断是否为 ATX 标题
        match_atx = atx_pattern.match(line)
        if match_atx:
            # group(1) 例如 "##"
            # group(2) 例如 "3.1 Introduction"
            hashes = match_atx.group(1)
            heading_text_raw = match_atx.group(2).strip()
            
            # 计算标题层级: "# -> level=0, ## -> level=1, ### -> level=2"
            heading_level = len(hashes) - 1

            # 移除类似 "3.1.3 " 的前缀
            heading_text = leading_numbers_pattern.sub('', heading_text_raw).strip()

            i += 1  # 跳过标题行，准备收集正文
            content_string, new_idx = gather_content(i)

            sections.append((heading_level, heading_text, content_string))
            i = new_idx
            print(heading_level, heading_text)
        else:
            # 否则跳到下一行
            i += 1
        
        # [可选调试输出] 打印当前标题层级及其文本
        

    return sections[1:]
    
def abstract_to_tex(section):
    """
    将 Markdown 中的 abstract 段落转化为 LaTeX 片段。

    参数:
        section: (level, heading_text, content_string)
            level: 0 表示一级标题, 1 表示二级标题, etc.
            heading_text: 当前标题文字
            content_string: 该标题下的 Markdown 文本
    
    返回:
        一个字符串，包含对应的 LaTeX abstract 环境。
    """
    level, heading_text, content_string = section

    # 如果标题不是 "Abstract"，则直接返回空字符串
    if heading_text.lower() != "abstract":
        return ""

    # 生成 LaTeX abstract 环境
    latex_abstract = (
        "\\begin{abstract}\n"
        f"{content_string}\n"
        "\\end{abstract}"
    )
    return latex_abstract

def references_to_tex(section):
    """
    将 Markdown 中的 references 段落转化为 LaTeX 片段。

    参数:
        section: (level, heading_text, content_string)
            level: 0 表示一级标题, 1 表示二级标题, etc.
            heading_text: 当前标题文字
            content_string: 该标题下的 Markdown 文本
    
    返回:
        一个字符串，包含对应的 LaTeX references 环境。
    """
    level, heading_text, content_string = section

    # 如果标题不是 "References"，则直接返回空字符串
    if heading_text.lower() != "references":
        return ""

    # 在每一行的末尾添加 \\ 以实现换行
    lines = content_string.splitlines()
    latex_content = " \\\\{}\n".join(line.strip() for line in lines if line.strip())

    # 生成 LaTeX 片段，使用 \section* 创建不带编号的标题
    latex_references = (
        "\\section*{References}\n"  # 不带编号的 section
        f"{latex_content}"
    )
    return latex_references

def md_to_tex_section(section):

    """
    将单个 Markdown 分段 (level, heading, content) 转化为 LaTeX 片段。
    会根据标题的深度生成 \\section, \\subsection, 或 \\subsubsection 等。
    同时对 markdown 中的图片 div 进行正则替换，转化为 LaTeX figure 环境。
    
    参数:
        section: (level, heading_text, content_string)
            level: 0 表示一级标题, 1 表示二级标题, etc.
            heading_text: 当前标题文字
            content_string: 该标题下的 Markdown 文本
    
    返回:
        一个字符串，包含对应的 LaTeX 标题以及内容。
        内容由 OpenAI 模型将 Markdown 转为 LaTeX，并将图片 div 转为 LaTeX figure。
    """
    level, heading_text, content_string = section
    
    # 根据 heading level 生成对应的 LaTeX 命令
    if level == 0:
        latex_heading = f"\\section{{{heading_text}}}"
    elif level == 1:
        latex_heading = f"\\subsection{{{heading_text}}}"
    elif level == 2:
        latex_heading = f"\\subsubsection{{{heading_text}}}"
    else:
        # 更深入的层级可自行添加
        latex_heading = f"\\paragraph{{{heading_text}}}"
    
    # 先粗略替换图片 div 为占位符，后续交由 OpenAI 模型或自身再做处理
    # 这里我们先把 <div style="text-align:center">...<img ...>...</div><div ...>Fig x: ...</div> 转换为一个自定义标记 [IMG_BLOCK] ... [END_IMG_BLOCK]
    # 这样后面可以更好控制让 OpenAI 转成正确的 LaTeX 也行，或在本地处理也行。
    # 这里我们本地进行处理，将它直接转换为 LaTeX figure。
    
    def replace_img_div(match):
        """
        将 <div style="text-align:center"> <img src="..." alt="..." style="width:60%;"/> </div>
        <div style="text-align:center;font-size:smaller;">Fig x: ...</div>
        这种模式转换为标准 LaTeX figure 环境
        """
        whole_block = match.group(0)
        
        # 提取 src
        src_match = re.search(r'<img.*?src="(.*?)".*?>', whole_block, re.DOTALL)
        src_path = src_match.group(1) if src_match else "image_not_found"
        
        # 提取 alt
        alt_match = re.search(r'<img.*?alt="(.*?)".*?>', whole_block, re.DOTALL)
        alt_text = alt_match.group(1) if alt_match else ""
        
        # 提取 caption (Fig x: ...)
        fig_match = re.search(r'Fig\s*\d+:\s*(.*?)<\/div>', whole_block, re.DOTALL)
        fig_caption = fig_match.group(1).strip() if fig_match else ""
        
        # 生成 LaTeX figure
        latex_figure = (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=0.6\\textwidth]{{{src_path}}}\n"
            f"  \\caption{{{alt_text if alt_text else fig_caption}}}\n"
            # 也可以根据需求决定是否加 label
            "\\end{figure}\n"
        )
        
        return latex_figure
    
    # 用正则定位该模式并转换为 latex figure
    # 该模式大概是:
    # <div style="text-align:center">.*?<img src="...".*?>.*?</div>\s*<div style="text-align:center;font-size:smaller;">.*?</div>
    # 这里用非贪婪模式, DOTALL 允许匹配换行
    pattern_img_div = re.compile(
        r'<div\s+style="text-align:center".*?>.*?<img.*?>.*?</div>\s*<div\s+style="text-align:center;font-size:smaller;">.*?<\/div>',
        re.DOTALL
    )
    
    content_converted_images = re.sub(pattern_img_div, replace_img_div, content_string)
    
    # ------------------------------------------------
    # 调用 OpenAI 接口，将 (转换好图片 div 的) Markdown 转为 LaTeX
    # ------------------------------------------------
    system_prompt = (
        "You are a helpful assistant that converts Markdown text to rigorous LaTeX. "
        "Maintain inline formatting like bold, italics, and code blocks when possible. "
        "Simply format horizontally aligned text, lists, tables, etc. into valid LaTeX."
        "Use [LaTeX] ... [/LaTeX] to wrap the final content without the \\section\{\}."
        "If the content is mathematically descriptive, please insert exactly one LaTeX math equation with explaination ($...$)to describe it."
        "Do not include any other irrelevant information."
        "Remember to clean the refs such as \[1], \[2], \[3] inside the text to strip the backslashes to [1], [2], [3]. No any extra backslashes."
    )
    
    user_prompt = (
        "Convert the following Markdown content to LaTeX. The text may already contain "
        "some partial LaTeX for figures:\n\n"
        f"{content_converted_images}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # 从环境变量中获取 openai key 和 base url
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    
    # 初始化 Client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    chat_response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        max_tokens=32768,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=messages,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    
    # Stream the response
    tex_body = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            tex_body += chunk.choices[0].delta.content
    
    # 假设我们想在聊天回复中，使用 [LaTeX] ... [/LaTeX] 包裹最终内容，类似:
    # [LaTeX]
    # 你要的 tex body ...
    # [/LaTeX]
    
    # 可以用正则截取中间内容:
    pattern = r'\[LaTeX\](.*?)\[/LaTeX\]'
    match = re.search(pattern, tex_body, re.DOTALL)
    if match:
        # 如果拿到中间的内容 就用它, 否则就用全部
        tex_body = match.group(1).strip()
    
    # 去掉多余的空白
    tex_body = re.sub(r'\s+', ' ', tex_body).strip()
    
    # 整合 LaTeX 标题和转好的正文
    final_tex_snippet = latex_heading + "\n\n" + tex_body + "\n"
    print("Tex snippet:")
    print(final_tex_snippet)
    return final_tex_snippet

def md_to_tex_section_without_jpg(section):
    """
    将单个 Markdown 分段 (level, heading_text, content_string) 转化为 LaTeX 片段，
    不处理任何 HTML 或图片 div，仅调用 OpenAI 模型将普通 Markdown 转为 LaTeX。

    参数:
        section: (level, heading_text, content_string)
            - level: 0 表示一级标题, 1 表示二级标题, 2 表示三级标题等
            - heading_text: 当前标题文字
            - content_string: 该标题下的 Markdown 文本

    返回:
        一个字符串，包含对应的 LaTeX 标题以及转换后的正文。
    """

    level, heading_text, content_string = section
    
    # 1) 根据 level 生成对应的 LaTeX 命令
    #    你也可以改成更灵活的逻辑，比如多级。
    if level == 0:
        latex_heading = f"\\section{{{heading_text}}}"
    elif level == 1:
        latex_heading = f"\\subsection{{{heading_text}}}"
    elif level == 2:
        latex_heading = f"\\subsubsection{{{heading_text}}}"
    else:
        latex_heading = f"\\paragraph{{{heading_text}}}"

    # 2) 判断是否要跳过 LLM 转换
    #    这里给出几种常见原因：
    #    - 内容字符串为空或全是空白 (content_string.strip() == "")
    #    - 标题看起来只是一个段落号, 形如"3"、"3.1"、"3.1.1" 等 (可根据需要调宽或调窄判断规则)

    # 例：用一个正则匹配 `数字(.数字)*`，可带可不带后缀空格
    #   如果 heading_text 完全匹配这个模式，就认为它是个“纯编号标题”，不必调用 LLM
    pure_number_pattern = re.compile(r'^\d+(\.\d+)*$')

    # 先去一下两端空格
    ht_stripped = heading_text.strip()
    # 若正文为空，或标题是纯数字/编号，就跳过 LLM
    skip_llm = (not content_string.strip()) or bool(pure_number_pattern.match(ht_stripped))

    if skip_llm:
        # 直接返回标题 + 原始正文 (若有也可保留)
        # 如果你只想输出标题，就让正文为空
        tex_body = content_string
        # 也可以选择把正文丢弃，比如:
        # tex_body = ""
    else:
        # 3) 需要调用 LLM 的情况
        # system_prompt = (
        #     "You are a helpful assistant that converts Markdown text to rigorous LaTeX. "
        #     "Maintain inline formatting like bold, italics, and code blocks when possible. "
        #     "Simply format horizontally aligned text, lists, tables, etc. into valid LaTeX."
        #     "Use [LaTeX] ... [/LaTeX] to wrap the final content without the \\section\{\}."
        #     "If the content is mathematically descriptive, please insert exactly one LaTeX math equation with explaination (\\[...\\])to describe it."
        #     "You are forced to use \\begin{dmath} and \\end{dmath} to replace the origin square brackets and wrap the equation"
        #     "Do not include any other irrelevant information."
        #     "Remember to clean the refs such as \[1], \[2], \[3] inside the text to strip the backslashes to [1], [2], [3]. No any extra backslashes."
        # )
        system_prompt = (
            "You are a helpful assistant that converts Markdown text to rigorous LaTeX. "
            "Maintain inline formatting like bold, italics, and code blocks when possible. "
            "Format horizontally aligned text, lists, and tables into valid LaTeX.\n\n"

            "Use [LaTeX] ... [/LaTeX] to wrap the final content without the \\section{}.\n\n"
            "If the content is mathematically descriptive, please insert exactly one LaTeX math equation to describe it."
            "For mathematical content, strictly follow the **standard equation format** below:\n\n"

            "1. **Wrap equations inside `equation`**:\n"
            "   ```latex\n"
            "   \\begin{equation}\n"
            "       \\resizebox{0.95\\columnwidth}{!}{$\n"
            "       ...  % (Insert the equation here)\n"
            "       $}\n"
            "   \\end{equation}\n"
            "   ```\n"
            "   - **All equations must be enclosed in `\\resizebox{0.95\\columnwidth}{!}{...}`**.\n"
            "   - **Ensure the equation fits within `\\columnwidth`** in two-column layouts.\n\n"

            "2. **For descriptions, simply use plain text with double backslashes, for example:\n"
            "$f_i(x)$ is the local objective function of node $i$.\\"
            "$\mathcal{N}_i$ is the set of in-neighbors of node $i$.\\"

            "3. **Ensure proper formatting**:\n"
            "   - **DO NOT use `align`, `multline`, or `split`**—only `equation` with `resizebox`.\n"
            "   - **DO NOT allow formulas to exceed column width**.\n"
            "   - **DO NOT allow any other latex syntax such as" 
            "    \\documentclass{article} \\usepackage{amsmath} \\usepackage{graphicx} \\begin{document}** use the plain content with formula.\n"

            "   - **Maintain the original refs and ensure that references like [1], [2], [3], do not contain unnecessary backslashes**.\n\n"

            "All generated LaTeX content **must strictly adhere to this structure**."
        )

        user_prompt = (
            "Convert the following Markdown content to LaTeX. "
            f"{content_string}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 从环境变量中获取 openai key 和 base url
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")

        # 初始化 Client
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        chat_response = client.chat.completions.create(
            model=os.environ.get("MODEL"),
            max_tokens=32768,
            temperature=0.5,
            stop="<|im_end|>",
            stream=True,
            messages=messages,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        chat_response = client.chat.completions.create(
            model=os.environ.get("MODEL"),
            max_tokens=32768,
            temperature=0.5,
            stop="<|im_end|>",
            stream=True,
            messages=messages,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        # 流式读取返回
        tex_body = ""
        for chunk in chat_response:
            if chunk.choices[0].delta.content:
                tex_body += chunk.choices[0].delta.content

        # 提取 [LaTeX] ... [/LaTeX] 中间的内容
        pattern = r'\[LaTeX\](.*?)\[/LaTeX\]'
        match = re.search(pattern, tex_body, re.DOTALL)
        if match:
            tex_body = match.group(1).strip()

        # 去掉多余的空白
        tex_body = re.sub(r'\s+', ' ', tex_body).strip()

    # 4) 最终拼接
    final_tex_snippet = latex_heading + "\n\n" + tex_body + "\n"
    print("Tex snippet:")
    print(final_tex_snippet)
    return final_tex_snippet

def insert_section(tex_path: str, section_content: str):
    """
    将 section_content 追加到 .tex 文件“最后一个 section(或子节)的正文末尾”。
    具体逻辑如下：
      1. 如果文件内找不到任何 \section{...}、\subsection{...}、\subsubsection{...}，
         那么就将 section_content 插入到 \end{abstract} 之后。
      2. 如果在全文中能找到若干标题 (\section、\subsection、\subsubsection)，
         则将 section_content 插入到最后出现的那个标题对应正文的末尾（即它和下一个标题/文件结束之间）。
      3. 如果既没有 abstract 环境，也没有任何标题，则在 \end{document} 前插入。
    
    参数:
        tex_path: str
            .tex 文件的路径。
        section_content: str
            需要插入的段落字符串（LaTeX 格式）。
    
    注意：
        - 这段逻辑会将新的内容**追加**到最后一个标题所对应正文的末尾，
          这样可以避免把之前的内容“分割”或“顶开”。
    """

    if not os.path.exists(tex_path):
        print(f"TeX 文件不存在: {tex_path}")
        return

    with open(tex_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 正则匹配标题、abstract、document
    # 注意 \section、\subsection、\subsubsection 都做单独分组，这样获取行号时好区分
    title_pattern = re.compile(r'^(\\section|\\subsection|\\subsubsection)\{[^}]*\}')
    end_abstract_pattern = re.compile(r'^\\end\{abstract\}')
    end_document_pattern = re.compile(r'^\\end\{document\}')

    # 找到所有标题行号，保存到列表
    title_lines = []
    end_abstract_line = None
    end_document_line = None

    for i, line in enumerate(lines):
        if title_pattern.match(line.strip()):
            title_lines.append(i)
        elif end_abstract_pattern.match(line.strip()):
            end_abstract_line = i
        elif end_document_pattern.match(line.strip()):
            end_document_line = i

    # 将要插入的内容行列表
    insert_content_lines = section_content.strip().split('\n')

    # 如果找不到任何标题
    if not title_lines:
        # 如果有 \end{abstract}，就插在 \end{abstract} 后
        if end_abstract_line is not None:
            insert_idx = end_abstract_line + 1
        else:
            # 没有 \end{abstract}，就尝试在 \end{document} 之前插入
            if end_document_line is not None:
                insert_idx = end_document_line
            else:
                # 如果也没有 \end{document}，就插到文件末尾
                insert_idx = len(lines)

        new_lines = (
            lines[:insert_idx]
            + [l + "\n" for l in insert_content_lines]
            + lines[insert_idx:]
        )

    else:
        # 有标题时，将内容追加到“最后一个标题对应正文”的末尾
        last_title_line = title_lines[-1]

        # 找到下一个标题的行号（如果有），或 \end{document} 行号，以确定正文区间结束
        # “最后标题正文”从 last_title_line+1 一直到 next_title_line-1（或结束）
        next_boundaries = [end_document_line if end_document_line is not None else len(lines)]
        for t_line in title_lines:
            if t_line > last_title_line:
                next_boundaries.append(t_line)
        # next_boundary 是最后标题之后遇到的第一个 boundary（若没有, 就是文件末尾）
        next_boundary = min(next_boundaries) if next_boundaries else len(lines)

        # 我们希望将新的内容插在“最后标题正文的最末尾”之后，也就是说在 next_boundary 前。
        # 不过若“最后标题”本身就处于全文件最终，next_boundary 可能表示文件末尾/文档结束。
        # 这里为了避免把最后一行顶下去，可以先把其中的正文行都保留，再在最后插入 section_content。
        new_lines = []
        new_lines.extend(lines[:next_boundary])  # 保留从头到最后正文结束
        new_lines.extend([l + "\n" for l in insert_content_lines])
        new_lines.extend(lines[next_boundary:])

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("成功插入 section 内容:", tex_path)

def md_to_tex(md_path, tex_path, title):
    """
    将 Markdown 文件转换为 LaTeX 文件。

    参数:
        md_path (str): 输入的 Markdown 文件路径。
        tex_path (str): 输出的 LaTeX 文件路径。
    """
    sections = search_sections(md_path)
    section_index = 0
    while section_index < len(sections):
        print(f"Converting section {section_index+1}/{len(sections)}")
        if section_index == 0:
            tex = abstract_to_tex(sections[section_index])
            print(tex)
        elif section_index == len(sections) - 1:
            postprocess(tex_path, title)
            tex = references_to_tex(sections[section_index])
            print(tex)
        else:
            tex = md_to_tex_section_without_jpg(sections[section_index])
            print(tex)
        insert_section(tex_path, tex)
        section_index += 1
    # tex_to_pdf(tex_path, output_dir=os.path.dirname(tex_path), compiler="pdflatex")

def tex_to_pdf(tex_path, output_dir=None, compiler="xelatex"):
    """
    将 LaTeX 文件编译为 PDF 文件。

    参数:
        tex_path (str): 输入的 LaTeX 文件路径。
        output_dir (str): 输出的 PDF 文件目录。
        compiler (str): 编译器，默认为 "xelatex"。
    """
    if output_dir is None:
        output_dir = os.path.dirname(tex_path)
    tex_name = os.path.basename(tex_path)
    tex_name_no_ext = os.path.splitext(tex_name)[0]
    pdf_path = os.path.join(output_dir, f"{tex_name_no_ext}.pdf")
    
    subprocess.run([
        compiler,
        "-interaction=nonstopmode",
        "-output-directory",
        output_dir,
        tex_path
    ])
    
    print(f"PDF 文件已生成: {pdf_path}")

def insert_figures(png_path, tex_path, json_path, ref_names, survey_title, new_tex_path):
    """
    读取给定的 TeX 文件 (tex_path)，先调用 insert_outline_figure 在其中插入概览图片；
    然后再调用 insert_tex_images 在文中发现的引用标记位置插入 figure 环境。
    最后把处理完的文本写入 new_tex_path。
    
    参数：
      png_path:       大纲图片的路径（会传给 insert_outline_figure）。
      tex_path:       原始 TeX 文件路径。
      json_path:      图片对应的 JSON（会传给 insert_tex_images，内含 引用名称 -> 图片路径 的映射）。
      ref_names:      引用名称列表 (index 从 0 开始)。
      survey_title:   用于大纲图片 figure 中的说明文字。
      new_tex_path:   处理后新的 TeX 文件输出路径。
    """
    # 1. 读取原始 tex 文件内容
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()

    # 2. 在 '2 Introduction' 前插入一张占满整页的描述性图片（概览图）
    updated_tex = insert_outline_figure(
        png_path=png_path,
        tex_content=tex_content,
        survey_title=survey_title
    )

    # 3. 在文中其他引用 [n], \[n], \[n\] 等位置插入 figure
    updated_tex = insert_tex_images(
        json_path=json_path,
        ref_names=ref_names,
        text=updated_tex
    )

    # 4. 将处理结果写入新路径
    with open(new_tex_path, 'w', encoding='utf-8') as f:
        f.write(updated_tex)

    print(f"已生成新的 TeX 文件: {new_tex_path}")
    return new_tex_path

def postprocess(tex_path, new_title):
    """
    读取给定的 TeX 文件 (tex_path):
      1) 在第一处 \author 行的上一行插入 \title{new_title}。
      2) 将所有形如 "\[1\]"、"\[1]"、以及 "\[12\]" 等引用标记，
         以及 "[1\]" 之类的混合形式，全都去掉反斜杠，统一替换为 [1]、[12]。
      3) 将所有由 \[ \] 包裹的数学公式都替换为 \begin{dmath} \end{dmath}。
    最后将结果覆盖写回原始文件，并返回 tex_path。
    """
    new_title = 'A Survey of ' + new_title
    # 1) 读取文件行
    with open(tex_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2) 找到包含 "\author" 的行，在其上一行插入 \title{...}
    inserted = False
    for i, line in enumerate(lines):
        # 如果这一行以 \author 开头，或包含 \author{...} 等
        if line.strip().startswith(r'\author'):
            # 在它前面插入一行 \title{new_title}
            lines.insert(i, f'\\title{{{new_title}}}\n')
            inserted = True
            break

    if not inserted:
        # 若整份文件都没有 \author{...}，可以选择直接在文档末尾插入，或在开头插入，视需求而定
        print(f"[警告] 未找到 '\\author' 行，未能插入 '\\title{{{new_title}}}'。")
        # 下面演示直接在文档末尾插入（可按自己需求改放开头等）
        lines.append(f'\\title{{{new_title}}}\n')

    # 将行列表合并为一个整体字符串，便于进行正则替换
    text_joined = ''.join(lines)

    # 3) 将形如 "\[1\]"、"\[12]"、"[12\]" 等都换成 "[1]"、"[12]" 等
    #    核心正则：'(?:\\)?\[(\d+)(?:\\)?\]'
    #      (?:\\)? ---- 可选的一个反斜杠
    #      \[      ---- 匹配方括号的开头 '['
    #      (\d+)   ---- 匹配并捕获1--多位数字
    #      (?:\\)? ---- 可选的一个反斜杠
    #      \]      ---- 匹配方括号的结尾 ']'
    ref_pattern = re.compile(r'(?:\\)?\[(\d+)(?:\\)?\]')
    text_processed = ref_pattern.sub(r'[\1]', text_joined)

    # 4) 将所有由 \[ \] 包裹的数学公式替换为 \begin{dmath} \end{dmath}
    #    正则示例: 匹配 \[ ... \] 中间任意内容 (非贪婪)
    #    使用 DOTALL 选项让 '.' 匹配换行
    # eq_pattern = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
    # text_processed = eq_pattern.sub(r'\\begin{dmath}\1\\end{dmath}', text_processed)

    # 5) 写回原文件
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(text_processed)

    print(f"[完成] 已在 '{tex_path}' 中插入/追加 \\title{{{new_title}}}，替换引用标记并将公式转为 dmath 格式。")
    return tex_path

def md_to_tex_to_pdf(md_path, tex_path, pdf_path, png_path, json_path, ref_names, survey_title):
    """
    将 Markdown 文件转换为 LaTeX 文件，然后再编译为 PDF 文件。

    参数:
        md_path (str): 输入的 Markdown 文件路径。
        tex_path (str): 输出的 LaTeX 文件路径。
        pdf_path (str): 输出的 PDF 文件路径。
    """
    md_to_tex(md_path, tex_path)
    new_tex_path = insert_figures(png_path, tex_path, json_path, ref_names, survey_title, tex_path)
    # tex_to_pdf(new_tex_path, output_dir=os.path.dirname(tex_path), compiler="pdflatex")

if __name__ == "__main__":
    # 读取环境变量 
    dotenv.load_dotenv()
    # md_path = preprocess_md("src/demo/latex_template/test copy.md", "src/demo/latex_template/test_preprocessed.md")
    md_path = 'src/static/data/info/undefined/survey_undefined_preprocessed.md'
    tex_path = "src/static/data/info/undefined/template.tex"
    md_to_tex(md_path, tex_path, title="A Comprehensive Review of ADMM On Consensus Distributed Optimization")
    # insert_figures('src/static/data/info/undefined/outline.png', 
    #                'src/demo/latex_template/template.tex', 
    #                'src/static/data/info/undefined/flowchart_results.json', 
    #                ['A comprehensive review of recommender systems transitioning from theory to practice', 'A large language model enhanced conversational recommender system'],
    #                'Survey Title',
    #                'src/demo/latex_template/template_with_figures.tex')
    tex_to_pdf(tex_path, output_dir=os.path.dirname(tex_path), compiler="xelatex")
