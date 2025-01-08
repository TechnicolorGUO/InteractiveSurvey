from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import transformers
import ast
import uuid
import re
import os
import json
import chromadb
import time
import openai
import dotenv
import json
import base64
import concurrent.futures
import numpy as np
from numpy.linalg import norm
from .asg_retriever import Retriever

def getQwenClient(): 
    # openai_api_key = os.environ.get("OPENAI_API_KEY")
    # openai_api_base = os.environ.get("OPENAI_API_BASE")
    openai_api_key = "qwen2.5-72b-instruct-8eeac2dad9cc4155af49b58c6bca953f"
    openai_api_base = "https://its-tyk1.polyu.edu.hk:8080/llm/qwen2.5-72b-instruct"
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    return client

def generateResponse(client, prompt):
    chat_response = client.chat.completions.create(
        model="Qwen2.5-72B-Instruct",
        max_tokens=768,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=[{"role": "user", "content": prompt}]
    )
    # Stream the response to console
    text = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    return text

def generateResponseIntroduction(client, prompt):
    chat_response = client.chat.completions.create(
        model="Qwen2.5-72B-Instruct",
        max_tokens=1024,
        temperature=0.7,
        stop="<|im_end|>",
        stream=True,
        messages=[{"role": "user", "content": prompt}]
    )
    # Stream the response to console
    text = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    return text

def generate_introduction(context, client):
    """
    Generates an introduction based on the context of a survey paper.
    The introduction is divided into four parts: 
    1. Background of the general topic.
    2. Main problems mentioned in the paper.
    3. Contributions of the survey paper.
    4. The aim of the survey paper. 
    Total length is limited to 500-700 words.
    """

    template = '''
Directly generate an introduction based on the following context (a survey paper). 
The introduction includes 4 elements: background of the general topic (1 paragraph), main problems mentioned in the paper (1 paragraph), contributions of the survey paper (2 paragraphs), and the aim and structure of the survey paper (1 paragraph). 
The introduction should strictly follow the style of a standard academic introduction, with the total length of 500-700 words. Do not include any headings (words like "Background:", "Problems:") or extra explanations except for the introduction and exclude all citations or references.

Context:
{context}

Introduction:
'''

    formatted_prompt = template.format(context=context)
    response = generateResponseIntroduction(client, formatted_prompt)

    # 从生成的结果中提取答案
    answer_start = "Introduction:"
    start_index = response.find(answer_start)
    if start_index != -1:
        answer = response[start_index + len(answer_start):].strip()
    else:
        answer = response.strip()

    # 将生成的引言分段
    paragraphs = answer.split("\n\n")
    if len(paragraphs) < 5:
        # 确保生成的引言包含五个段落
        answer = "\n\n".join(paragraphs[:5])
    else:
        answer = "\n\n".join(paragraphs)

    return answer

def generate_future_work(context, client):
    """
    Generates the Future Work section based on the context of a survey paper.
    The Future Work section is typically structured into:
    1. Summary of current limitations or gaps.
    2. Proposed directions for future research.
    3. Potential impact of the proposed future work.
    Total length is limited to 300-500 words.
    """
    
    template = '''
Directly generate the Future Work section based on the following context (a survey paper). 
The section includes 3 elements:
1. Summary of current limitations or gaps (1 paragraph).
2. Proposed directions for future research (1-2 paragraphs).
3. Potential impact of the proposed future work (1 paragraph).
The Future Work section should strictly follow the style of a standard academic paper, with a total length of 300-500 words. Do not include any headings or extra explanations except for the Future Work content and exclude all citations or references.

Context:
{context}

Future Work:
'''
    
    formatted_prompt = template.format(context=context)
    response = generateResponseIntroduction(client, formatted_prompt)  # Assuming this function handles the response generation
    
    # Extract the Future Work content from the response
    answer_start = "Future Work:"
    start_index = response.find(answer_start)
    if start_index != -1:
        answer = response[start_index + len(answer_start):].strip()
    else:
        answer = response.strip()
    
    # Split the content into paragraphs
    paragraphs = answer.split("\n\n")
    if len(paragraphs) < 3:
        # Ensure the Future Work section has at least three paragraphs
        answer = "\n\n".join(paragraphs[:3])
    else:
        answer = "\n\n".join(paragraphs)
    
    return answer

def generate_conclusion(context, client):
    """
    Generates the Conclusion section based on the context of a survey paper.
    The Conclusion is typically structured into:
    1. Recap of the main findings or discussions.
    2. Significance of the survey.
    3. Final remarks or call to action.
    Total length is limited to 300-500 words.
    """
    
    template = '''
Directly generate the Conclusion section based on the following context (a survey paper). 
The section includes 3 elements:
1. Recap of the main findings or discussions (1 paragraph).
2. Significance of the survey (1 paragraph).
3. Final remarks or call to action (1 paragraph).
The Conclusion should strictly follow the style of a standard academic paper, with a total length of 300-500 words. Do not include any headings or extra explanations except for the Conclusion content and exclude all citations or references.

Context:
{context}

Conclusion:
'''
    
    formatted_prompt = template.format(context=context)
    response = generateResponseIntroduction(client, formatted_prompt)  # Assuming this function handles the response generation
    
    # Extract the Conclusion content from the response
    answer_start = "Conclusion:"
    start_index = response.find(answer_start)
    if start_index != -1:
        answer = response[start_index + len(answer_start):].strip()
    else:
        answer = response.strip()
    
    # Split the content into paragraphs
    paragraphs = answer.split("\n\n")
    if len(paragraphs) < 3:
        # Ensure the Conclusion section has at least three paragraphs
        answer = "\n\n".join(paragraphs[:3])
    else:
        answer = "\n\n".join(paragraphs)
    
    return answer

def generate_introduction_alternate(title, context, client):

    template_explicit_section = '''
Directly generate an introduction based on the following context (a survey paper).
The introduction should include six parts: 
1. Background of the general topic (1 paragraph).
2. The research topic of this survey paper (1 paragraph).
3. A summary of the first section following the Introduction (1 paragraph).
4. A summary of the second section following the Introduction (1 paragraph).
5. A summary of the third section following the Introduction (1 paragraph).
6. Contributions of the survey paper (1 paragraph).
The introduction should strictly follow the style of a standard academic introduction, with the total length limited to 600-700 words. Do not include any headings (words like "Background:", "Summary:") or extra explanations except for the introduction.

Context:
{context}

Introduction:
'''

    template = '''
Directly generate an introduction based on the following context (a survey paper).
The introduction should include 4 parts (6 paragraphs): 
1. Background of the general topic (1 paragraph).
2. The research topic of this survey paper (1 paragraph).
3. A detailed overview of the content between the Introduction and Future direction / Conclusion (3 paragraphs).
4. Contributions of the survey paper (1 paragraph).
The introduction should strictly follow the style of a standard academic introduction, with the total length limited to 600-700 words. Do not include any headings (words like "Background:", "Summary:") or extra explanations except for the introduction.
Do not include any sentences like "The first section...", "The second section...".


Survey title:
{title}
Context:
{context}

Introduction:
'''

    formatted_prompt = template.format(context=context, title=title)
    response = generateResponseIntroduction(client, formatted_prompt)

    # 从生成的结果中提取答案
    answer_start = "Introduction:"
    start_index = response.find(answer_start)
    if start_index != -1:
        answer = response[start_index + len(answer_start):].strip()
    else:
        answer = response.strip()

    # 将生成的引言分段
    paragraphs = answer.split("\n\n")
    if len(paragraphs) < 6:
        # 确保生成的引言包含六个段落
        answer = "\n\n".join(paragraphs[:6])
    else:
        answer = "\n\n".join(paragraphs)

    return answer

# Parse the outline and filter sections/subsections for content generation
def parse_outline_with_subsections(outline):
    """
    解析 outline 去掉一级标题并根据二级和三级标题的结构进行筛选。
    """
    outline_list = ast.literal_eval(outline)
    selected_subsections = []

    # 遍历 outline 的每一部分，生成对应的内容
    for i, (level, section_title) in enumerate(outline_list):
        if level == 1:
            continue  # 跳过一级标题

        elif level == 2:  # 如果是二级目录
            if i + 1 < len(outline_list) and outline_list[i + 1][0] == 3:
                # 三级目录存在，跳过当前二级标题
                continue
            else:
                selected_subsections.append((level, section_title))  # 处理没有内层的二级 section

        elif level == 3:  # 处理三级目录
            selected_subsections.append((level, section_title))

    return selected_subsections

def process_outline_with_empty_sections(outline_list, selected_outline, context, client):
    content = ""
    
    # 遍历原始 outline 的每一部分，生成对应的内容
    for level, section_title in outline_list:
        # 如果在筛选出的部分里，生成内容；否则保留空的部分
        if (level, section_title) in selected_outline:
            if level == 1:
                content += f"# {section_title}\n"
            elif level == 2:
                content += f"## {section_title}\n"
            elif level == 3:
                content += f"### {section_title}\n"
            
            # 调用 LLM 生成内容
            section_content = generate_survey_section(context, client, section_title)
            content += f"{section_content}\n\n"
        else:
            # 生成空内容部分
            if level == 1:
                content += f"# {section_title}\n\n"
            elif level == 2:
                content += f"## {section_title}\n\n"
            elif level == 3:
                content += f"### {section_title}\n\n"
    
    return content

def process_outline_with_empty_sections_new(outline_list, selected_outline, context_list, client):
    content = ""
    context_dict = {title: ctx for (lvl, title), ctx in zip(selected_outline, context_list)}
    
    for level, section_title in outline_list:
        if (level, section_title) in selected_outline:
            if level == 1:
                content += f"# {section_title}\n"
            elif level == 2:
                content += f"## {section_title}\n"
            elif level == 3:
                content += f"### {section_title}\n"
            
            section_content = generate_survey_section(context_dict[section_title], client, section_title)
            content += f"{section_content}\n\n"
        else:
            if level == 1:
                content += f"# {section_title}\n\n"
            elif level == 2:
                content += f"## {section_title}\n\n"
            elif level == 3:
                content += f"### {section_title}\n\n"

    return content

def process_outline_with_empty_sections_new_new(outline_list, selected_outline, context_list, client):
    content = ""
    context_dict = {title: ctx for (lvl, title), ctx in zip(selected_outline, context_list)}
    
    # 准备需要生成内容的章节列表
    sections_to_generate = []
    for level, section_title in outline_list:
        if (level, section_title) in selected_outline:
            sections_to_generate.append((level, section_title))
    
    # 定义用于生成章节内容的函数
    def generate_section(section_info):
        level, section_title = section_info
        section_content = generate_survey_section(context_dict[section_title], client, section_title)
        return (section_title, level, section_content)
    
    # generate survey section in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_section = {executor.submit(generate_section, section): section for section in sections_to_generate}
        generated_sections = {}
        for future in concurrent.futures.as_completed(future_to_section):
            section = future_to_section[future]
            try:
                section_title, level, section_content = future.result()
                generated_sections[section_title] = (level, section_content)
            except Exception as exc:
                print(f'{section} generated an exception: {exc}')

    # combine generated sections with outline   
    for level, section_title in outline_list:
        if (level, section_title) in selected_outline:
            if level == 1:
                content += f"# {section_title}\n"
            elif level == 2:
                content += f"## {section_title}\n"
            elif level == 3:
                content += f"### {section_title}\n"
            
            # 添加生成的内容
            if section_title in generated_sections:
                section_content = generated_sections[section_title][1]
                content += f"{section_content}\n\n"
            else:
                content += "\n\n"
        else:
            if level == 1:
                content += f"# {section_title}\n\n"
            elif level == 2:
                content += f"## {section_title}\n\n"
            elif level == 3:
                content += f"### {section_title}\n\n"
    
    return content

# wza
def process_outline_with_empty_sections_citations(outline_list, selected_outline, context_list, client, citation_data_list):
    # 将selected_outline和context_list转成dict以根据section_title获取context
    context_dict = {title: ctx for (lvl, title), ctx in zip(selected_outline, context_list)}

    sections_to_generate = [(level, title) for (level, title) in outline_list if (level, title) in selected_outline]

    def generate_section_with_citations_wrapper(section_info):
        level, section_title = section_info
        section_context = context_dict[section_title]
        section_content = generate_survey_section_with_citations(section_context, client, section_title, citation_data_list)
        return section_title, level, section_content

    generated_sections = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_section = {executor.submit(generate_section_with_citations_wrapper, s): s for s in sections_to_generate}
        for future in concurrent.futures.as_completed(future_to_section):
            s_title, s_level, s_content = future.result()
            generated_sections[s_title] = (s_level, s_content)

    content = ""
    for level, section_title in outline_list:
        if section_title in generated_sections:
            s_level, s_content = generated_sections[section_title]
            if level == 1:
                content += f"# {section_title}\n{s_content}\n\n"
            elif level == 2:
                content += f"## {section_title}\n{s_content}\n\n"
            elif level == 3:
                content += f"### {section_title}\n{s_content}\n\n"
        else:
            if level == 1:
                content += f"# {section_title}\n\n"
            elif level == 2:
                content += f"## {section_title}\n\n"
            elif level == 3:
                content += f"### {section_title}\n\n"
    return content


# wza
def generate_survey_section_with_citations_old(context, client, section_title, citation_data_list, 
                                           temp=0.5, base_threshold=0.7, dynamic_threshold=True):    
    template = """
Generate a detailed and technical content for a survey paper's section based on the following context.
The generated content should be in 3 paragraphs of no more than 300 words in total, following the style of a standard academic survey paper.
It is expected to dive deeply into the section title "{section_title}".
Directly return the 3-paragraph content without any other information and exclude all citations or references.

Context: 
{context}
------------------------------------------------------------
Survey Paper Content for "{section_title}":
"""
    formatted_prompt = template.format(context=context, section_title=section_title)
    response = generateResponse(client, formatted_prompt).strip()
    sentences = re.split(r'(?<=[.!?])\s+', response.strip())

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    sentence_embeddings = embedder.embed_documents(sentences)
    chunk_texts = [c["content"] for c in citation_data_list]
    chunk_sources = [c["source"] for c in citation_data_list]
    chunk_embeddings = embedder.embed_documents(chunk_texts)

    # 定义一个函数，用于计算两个向量之间的cosine similarity，避免分母为 0 加了 1e-9 做平滑
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a)*norm(b) + 1e-9)

    # 这里对每个句子的向量 s_emb，与所有引用块向量 c_emb 两两计算相似度，形成一行 row
    # 然后把所有 row 组成一个列表 sim_matrix，相当于一个二维矩阵
    sim_matrix = []
    for s_emb in sentence_embeddings:
        row = [cosine_sim(s_emb, c_emb) for c_emb in chunk_embeddings]
        sim_matrix.append(row)
    
    # 转成 Numpy
    sim_matrix = np.array(sim_matrix)

    # 将相似度矩阵展平后，计算 mean 和 std
    # 用 mean + k*std 和 base_threshold 取最大值作为动态 threshold
    # 否则直接用 base_threshold 作为阈值
    # 这个 threshold 用来判断句子与某段引用文本是否足够相似，才能打上引用
    all_sims = sim_matrix.flatten()
    mean = np.mean(all_sims)
    std = np.std(all_sims)
    k = 0.5
    threshold = max(base_threshold, mean + k*std) if dynamic_threshold else base_threshold

    # 这里遍历所有句子以及它们对应的引用相似度，如果相似度 >= threshold，就把 (句子ID, 引用块ID, 相似度) 放进 candidates
    candidates = []
    for i, sent in enumerate(sentences):
        for j, sim in enumerate(sim_matrix[i]):
            if sim >= threshold:
                candidates.append((i,j,sim))

    # 这段逻辑是为了保证至少有 min_references 条引用，但是 threshold -= 0.05 会导致部分section引用数量暴增
    # 如果引用不足，并且 threshold 还大于 0.1，就不断降低 threshold 以获得更多符合条件的引用
    # min_references = 0
    # current_refs = len(candidates)
    # while current_refs < min_references and threshold > 0.1:
    #     threshold -= 0.05
    #     candidates = []
    #     for i, sent in enumerate(sentences):
    #         for j, sim in enumerate(sim_matrix[i]):
    #             if sim >= threshold:
    #                 candidates.append((i,j,sim))
    #     current_refs = len(candidates)

    # 准备一个字典 source_count，用于统计每个文献 source 被使用了多少次
    source_count = {}
    for s in chunk_sources:
        source_count[s] = 0

    # 按照相似度由高到低对 candidates 排序，先分配高相似度的引用
    candidates.sort(key=lambda x: x[2], reverse=True)

    # assigned 是一个字典，将句子ID -> 文献source； diversity_limit 是对每个source的引用次数上限
    assigned = {}
    diversity_limit = 3

    # 逐个遍历排序后的 candidates如果当前句子(sent_id)还没有分配过引用，则获取对应文献source
    # 若该source在 source_count 中的使用次数还没达到 diversity_limit，就把这个 source 分配给该句子
    # 并且 source_count[src] 加1
    # 这样保证每个句子最多只加一个引用，并限制单个 source 不被滥用
    for (sent_id, chk_id, sim) in candidates:
        if sent_id not in assigned:
            src = chunk_sources[chk_id]
            if source_count[src] < diversity_limit:
                assigned[sent_id] = src
                source_count[src] += 1

    updated_sentences = []
    # 不使用编号，直接使用collection_name
    for i, sentence in enumerate(sentences):
        if i in assigned:
            collection_name = assigned[i]  # 直接使用source作为引用
            updated_sentences.append(sentence + f" [{collection_name}]")
        else:
            updated_sentences.append(sentence)

    updated_content = " ".join(updated_sentences)
    return updated_content

import re
import numpy as np
from numpy.linalg import norm
from langchain.embeddings import HuggingFaceEmbeddings

def generate_survey_section_with_citations(context, client, section_title, citation_data_list, 
                                           temp=0.5, base_threshold=0.7, dynamic_threshold=True):
    template = """
Generate a detailed and technical content for a survey paper's section based on the following context.
The generated content should be in 3 paragraphs of no more than 300 words in total, following the style of a standard academic survey paper.
It is expected to dive deeply into the section title "{section_title}".
Directly return the 3-paragraph content without any other information and exclude all citations or references.

Context: 
{context}
------------------------------------------------------------
Survey Paper Content for "{section_title}":
"""
    formatted_prompt = template.format(context=context, section_title=section_title)
    response = generateResponse(client, formatted_prompt).strip()
    
    # -- 1. 先将生成的文本按空行（即段落）拆分 ---
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    
    # -- 2. 拆分段落内部的句子，并记录每个句子所属的段落索引 ---
    all_sentences = []
    para_index_map = []  # 记录每个句子所属的段落编号
    for p_idx, para in enumerate(paragraphs):
        # 注意此处使用正则按 .!? 分句
        sentences_in_para = re.split(r'(?<=[.!?])\s+', para)
        for sent in sentences_in_para:
            if sent.strip():
                all_sentences.append(sent.strip())
                para_index_map.append(p_idx)

    # -- 3. 对所有句子进行向量化嵌入（保持逻辑：一次性处理全文） ---
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    sentence_embeddings = embedder.embed_documents(all_sentences)

    # -- 4. 对 citation_data_list 做向量化嵌入 ---
    chunk_texts = [c["content"] for c in citation_data_list]
    chunk_sources = [c["source"] for c in citation_data_list]
    chunk_embeddings = embedder.embed_documents(chunk_texts)

    # 定义一个函数计算余弦相似度
    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-9)

    # -- 5. 构建「句子-引用块」相似度矩阵 ---
    sim_matrix = []
    for s_emb in sentence_embeddings:
        row = [cosine_sim(s_emb, c_emb) for c_emb in chunk_embeddings]
        sim_matrix.append(row)
    sim_matrix = np.array(sim_matrix)

    # -- 6. 计算全局动态阈值（不按段落来，而是按全文来） ---
    all_sims = sim_matrix.flatten()
    mean_sim = np.mean(all_sims)
    std_sim = np.std(all_sims)
    k = 0.5
    threshold = max(base_threshold, mean_sim + k * std_sim) if dynamic_threshold else base_threshold

    # -- 7. 找出所有相似度 >= threshold 的 (句子ID, 引用块ID, 相似度) ---
    candidates = []
    for i, sent in enumerate(all_sentences):
        for j, sim in enumerate(sim_matrix[i]):
            if sim >= threshold:
                candidates.append((i, j, sim))

    # -- 8. 准备一个字典 source_count，用于统计每个文献 source 被使用了多少次 --
    source_count = {s: 0 for s in chunk_sources}
    # 对 candidates 按相似度降序排序
    candidates.sort(key=lambda x: x[2], reverse=True)

    # assigned 用来记录句子对应的引用文献
    assigned = {}
    diversity_limit = 3  # 同一个 source 最多被引用 3 次

    for (sent_id, chk_id, sim) in candidates:
        if sent_id not in assigned:
            src = chunk_sources[chk_id]
            if source_count[src] < diversity_limit:
                assigned[sent_id] = src
                source_count[src] += 1

    # -- 9. 将引用插入回句子 ---
    updated_sentences = []
    for i, sentence in enumerate(all_sentences):
        if i in assigned:
            updated_sentences.append(sentence + f" [{assigned[i]}]")
        else:
            updated_sentences.append(sentence)

    # -- 10. 将更新后的句子按原来的段落结构重新拼接 ---
    final_paragraphs = []
    current_p_idx = para_index_map[0] if para_index_map else 0
    temp_list = []

    for i, sent in enumerate(updated_sentences):
        p_idx = para_index_map[i]
        if p_idx != current_p_idx:
            # 说明进入了新的段落
            final_paragraphs.append(" ".join(temp_list))
            temp_list = []
            current_p_idx = p_idx
        temp_list.append(sent)

    # 别忘了把最后一个段落也加入
    if temp_list:
        final_paragraphs.append(" ".join(temp_list))

    # -- 11. 用空行重新拼接成多段落文本 ---
    updated_content = "\n\n".join(final_paragraphs)

    return updated_content


def generate_survey_section(context, client, section_title, temp=0.5):

    template = """
Generate a detailed and technical content for a survey paper's section based on the following context.
The generated content should be in 3 paragraphs of no more than 300 words in total, following the style of a standard academic survey paper.
It is expected to dive deeply into the section title "{section_title}".
Directly return the 3-paragraph content without any other information.

Context: 
{context}
------------------------------------------------------------
Survey Paper Content for "{section_title}":
"""

    formatted_prompt = template.format(context=context, section_title=section_title)
    response = generateResponse(client, formatted_prompt).strip()
    return response

# old
def generate_survey_paper_new(title, outline, context_list, client):
    parsed_outline = ast.literal_eval(outline)
    selected_subsections = parse_outline_with_subsections(outline)
    full_survey_content = process_outline_with_empty_sections_new_new(parsed_outline, selected_subsections, context_list, client)
    
    # Generate introduction and replace the existing one
    generated_introduction = generate_introduction_alternate(title, full_survey_content, client)
    introduction_pattern = r"(# 2 Introduction\n)(.*?)(\n# 3 )"
    full_survey_content = re.sub(introduction_pattern, rf"\1{generated_introduction}\n\3", full_survey_content, flags=re.DOTALL)
    return full_survey_content

# wza
def generate_survey_paper_new(title, outline, context_list, client, citation_data_list):
    parsed_outline = ast.literal_eval(outline)
    selected_subsections = parse_outline_with_subsections(outline)

    full_survey_content = process_outline_with_empty_sections_citations(
        parsed_outline,
        selected_subsections,
        context_list,
        client,
        citation_data_list
    )
    generated_introduction = generate_introduction_alternate(title, full_survey_content, client)
    introduction_pattern = r"(# 2 Introduction\n)(.*?)(\n# 3 )"
    full_survey_content = re.sub(introduction_pattern, rf"\1{generated_introduction}\n\3", full_survey_content, flags=re.DOTALL)
    return full_survey_content


def query_embedding_for_title(collection_name: str, title: str, n_results: int = 1, embedder: HuggingFaceEmbeddings = None):
    final_context = ""
    retriever = Retriever()
    title_embedding = embedder.embed_query(title)
    query_result = retriever.query_chroma(collection_name=collection_name, query_embeddings=[title_embedding], n_results=n_results)
    query_result_chunks = query_result["documents"][0]
    for chunk in query_result_chunks:
        final_context += chunk.strip() + "//\n"
    return final_context

def generate_context_list(outline, collection_list):
    context_list = []
    cluster_idx = -1
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    subsections = parse_outline_with_subsections(outline)
    for level, title in subsections:
        if(title.startswith("3")):
            cluster_idx = 0
        elif(title.startswith("4")):
            cluster_idx = 1
        elif(title.startswith("5")):
            cluster_idx = 2
        context_temp = ""
        for i in range(len(collection_list[cluster_idx])):
            context = query_embedding_for_title(collection_list[cluster_idx][i], title, embedder=embedder)
            context_temp += context
            context_temp += "\n"
        context_list.append(context_temp)
    print(f"Context list generated with length {len(context_list)}.")
    return context_list
