import json
import os
import re
from openai import OpenAI

from dotenv import load_dotenv
import requests
load_dotenv()

def getClient(): 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_base = os.environ.get("OPENAI_API_BASE")

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def generateResponse(client, prompt, max_tokens=768, temerature=0.5):
    chat_response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        max_tokens=max_tokens,
        temperature=temerature,
        stop="<|im_end|>",
        stream=True,
        messages=[{"role": "user", "content": prompt}]
    )

    text = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    return text

def robust_json_parse(raw_response):
    """
    Try to parse a JSON object from raw_response.
    If failed, try to extract the first {...} block and parse again.
    If still failed, return an empty dict and print a warning.
    """
    try:
        return json.loads(raw_response)
    except Exception:
        match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        # If all parsing fails, return empty dict and print warning
        print("Warning: Failed to parse LLM response as JSON, returning empty dict.\nRaw response:", raw_response)
        return {}
    
def download_arxiv_pdf(arxiv_id: str, save_dir: str) -> str:
    """
    下载 arXiv 论文 PDF 到指定目录
    :param arxiv_id: arXiv 主ID（如 '2301.00001'）
    :param save_dir: 目标文件夹路径（如 './pdfs'）
    :return: 保存后的PDF文件路径
    :raises: Exception 如果下载失败
    """
    # 构造PDF下载URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # 确保文件夹存在
    os.makedirs(save_dir, exist_ok=True)
    # 文件名
    file_path = os.path.join(save_dir, f"{arxiv_id}.pdf")

    # 如果已经存在，则跳过下载
    if os.path.exists(file_path):
        print(f"PDF already exists: {file_path}")
        return file_path

    print(f"Downloading {pdf_url} ...")
    try:
        resp = requests.get(pdf_url, stream=True, timeout=20)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Saved: {file_path}")
        return file_path
    except Exception as e:
        # 如果下载失败，尝试删除未完成的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Failed to download {arxiv_id}: {e}")
        raise

def extract_and_save_outline_from_md(md_file_path):
    if not os.path.isfile(md_file_path):
        raise FileNotFoundError(f"Markdown file not found: {md_file_path}")

    # 1. 读取md文件内容
    with open(md_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    outline = []
    pattern = r'^(#{1,6})\s+(.*)'

    # 2. 提取标题
    for line in lines:
        match = re.match(pattern, line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            outline.append([level, title])

    # 3. 生成json路径
    json_path = os.path.join(os.path.dirname(md_file_path), "outline.json")

    # 4. 保存为json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)
    
    # 5. 返回结果
    return outline

def extract_references_from_md(md_path):
    """
    Extract the References section from a Markdown file.
    Returns reference list as a string (one per line), or empty string if not found.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Match ## References 或 # Reference 及类似写法，允许多余空格或其他大小写
    pattern = re.compile(
        r'^(#{1,6})\s*(References?|Bibliography|参考文献)[\s#]*\n+([\s\S]*?)(?=^#{1,6}\s|\Z)', 
        re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(text)
    if match:
        references_block = match.group(3).strip()
        # 按行分割，去掉空行
        references = [line for line in references_block.splitlines() if line.strip()]
        return references
    else:
        return []

def fill_single_criterion_prompt(
    prompt_template: str,
    content: str,
    topic: str,
    criterion: dict,
    criteria_name: str,
    type: str
) -> str:
    """
    自动填充评价 prompt，支持 survey/outline，支持输出 json 格式
    :param prompt_template: prompt模板
    :param content: 需要评价的内容
    :param topic: 主题
    :param criterion: 评价标准
    :param criteria_name: 该 criterion 的名称
    :param content_key: 模板内容字段名
    :return: prompt字符串
    """
    format_args = {
        "topic": topic,
        "criterion_description": criterion['description'],
        "score_1": criterion['score 1'],
        "score_2": criterion['score 2'],
        "score_3": criterion['score 3'],
        "score_4": criterion['score 4'],
        "score_5": criterion['score 5'],
        "criteria_name": criteria_name,
        type: content
    }
    return prompt_template.format(**format_args)

def extract_topic_from_path(md_path: str) -> str:
    # 先绝对路径化，防止不同系统分隔符问题
    abs_path = os.path.abspath(md_path)
    # 获取上上级目录名
    topic = os.path.basename(os.path.dirname(os.path.dirname(abs_path)))
    return topic

def build_outline_tree_from_levels(outline_list):
    """
    将 [level, title] 格式的大纲解析为树结构。
    
    Args:
        outline_list: List of [level, title]，层数从1开始，顺序排列。

    Returns:
        node_objs: 所有节点字典（含children, parent, index）
        top_nodes: 顶层节点列表
    """
    node_objs = []
    for idx, (level, title) in enumerate(outline_list):
        node = {
            "level": level,
            "title": title,
            "index": idx,           # 唯一编号，用顺序号
            "children": [],
            "parent": None
        }
        node_objs.append(node)

    stack = []
    for node in node_objs:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            node["parent"] = stack[-1]["index"]
            stack[-1]["children"].append(node)
        stack.append(node)

    top_nodes = [node for node in node_objs if node["parent"] is None]
    return node_objs, top_nodes

def pdf2md(pdf_path):
    pass


if __name__ == "__main__":
    # 测试提取大纲
    md_file_path = "surveys/cs/3D Gaussian Splatting Techniques/AutoSurvey/3D Gaussian Splatting Techniques.md"
    outline = extract_and_save_outline_from_md(md_file_path)
    print(len(outline))
    for item in outline:
        print(item)
    # 打印大纲
    # print("Outline:")
    tree, top_nodes = build_outline_tree_from_levels(outline)
    print(len(tree))
