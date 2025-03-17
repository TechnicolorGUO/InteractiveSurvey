import re

def reindex_citations(content):
    """
    将content中所有形如[collection_name]的引用标记，全局重编号为[1]、[2]、[3]...。
    返回:
        new_content: 替换后的文本
        source_map: {collection_name: index, ...}
    """
    pattern = r"\[([^\[\]]+)\]"
    source_map = {}
    current_index = 1

    def replace_func(match):
        source = match.group(1)
        nonlocal current_index
        if source not in source_map:
            source_map[source] = current_index
            current_index += 1
        return f"[{source_map[source]}]"

    new_content = re.sub(pattern, replace_func, content)
    return new_content, source_map

def generate_references_section(source_map):
    """
    根据source_map生成References部分的文本。
    source_map: {collection_name: index, ...}

    返回值:
        str: 
        "References\n1 collection_name_1\n2 collection_name_2\n..."
    """
    # 将source_map按index排序
    index_to_source = sorted(source_map.items(), key=lambda x: x[1])
    refs_lines = ["References"]
    for source, idx in index_to_source:
        refs_lines.append(f"{idx} {source}")
    return "\n".join(refs_lines)
