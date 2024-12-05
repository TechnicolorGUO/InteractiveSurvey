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
The introduction should strictly follow the style of a standard academic introduction, with the total length of 500-700 words. Do not include any headings (words like "Background:", "Problems:") or extra explanations except for the introduction.

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
The Future Work section should strictly follow the style of a standard academic paper, with a total length of 300-500 words. Do not include any headings or extra explanations except for the Future Work content.

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
The Conclusion should strictly follow the style of a standard academic paper, with a total length of 300-500 words. Do not include any headings or extra explanations except for the Conclusion content.

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
    """
    Generate survey sections with citations using parallel processing.

    Args:
        outline_list (list): Parsed outline structure.
        selected_outline (list): List of selected sections/subsections.
        context_list (list): List of contexts for each section.
        client: LLM client for content generation.
        citation_data_list (list): Citation metadata for context chunks.

    Returns:
        str: Full survey content with citations.
    """
    content = ""
    context_dict = {title: (ctx, citations) for (lvl, title), ctx, citations in zip(selected_outline, context_list, citation_data_list)}

    # Function to generate content for a single section
    def generate_section_with_citations(section_info):
        level, section_title = section_info
        if section_title not in context_dict:
            return section_title, level, ""

        context, citation_data = context_dict[section_title]
        section_content = generate_survey_section_with_citations(context, client, section_title, citation_data)
        return section_title, level, section_content

    # Prepare sections to generate in parallel
    sections_to_generate = [(level, section_title) for level, section_title in outline_list if (level, section_title) in selected_outline]

    # Parallel processing for all sections
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_section = {executor.submit(generate_section_with_citations, section): section for section in sections_to_generate}
        generated_sections = {}
        for future in concurrent.futures.as_completed(future_to_section):
            section_title, level, section_content = future.result()
            generated_sections[section_title] = (level, section_content)

    # Combine results into final content
    for level, section_title in outline_list:
        if (level, section_title) in selected_outline:
            if level == 1:
                content += f"# {section_title}\n"
            elif level == 2:
                content += f"## {section_title}\n"
            elif level == 3:
                content += f"### {section_title}\n"

            # Add generated content
            if section_title in generated_sections:
                section_content = generated_sections[section_title][1]
                content += f"{section_content}\n\n"
            else:
                content += "\n\n"
        else:
            # Add empty section
            if level == 1:
                content += f"# {section_title}\n\n"
            elif level == 2:
                content += f"## {section_title}\n\n"
            elif level == 3:
                content += f"### {section_title}\n\n"

    return content

# wza
def generate_survey_section_with_citations(context, client, section_title, citation_data, 
                                           temp=0.5, base_threshold=0.5, dynamic_threshold=True):
    # Step 1: Generate content
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
    generated_content = generateResponse(client, formatted_prompt).strip()

    # Step 2: Split the generated content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', generated_content)

    # Step 3: Embed sentences
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    sentence_embeddings = embedder.embed_documents(sentences)

    # Step 4: Calculate threshold
    adjusted_threshold = base_threshold
    if dynamic_threshold:
        distances = [1 - citation.get("distance", 1.0) for citation in citation_data]
        if distances:
            avg_similarity = sum(distances) / len(distances)
            adjusted_threshold = min(max(avg_similarity * 0.9, 0.4), 0.85)

    # Step 5: Add citations
    updated_content = []
    for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
        relevant_sources = [citation["source"] for citation in citation_data 
                            if 1 - citation.get("distance", 1.0) >= adjusted_threshold]
        unique_citations = list(set(relevant_sources))
        citation_text = f" [{', '.join(unique_citations)}]" if unique_citations else ""
        updated_content.append(sentence + citation_text)

    # Combine sentences back into a single section
    return "\n\n".join(updated_content).strip()

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

# # wza
# def generate_survey_paper_new(title, outline, context_list, client, citation_data_list):
#     parsed_outline = ast.literal_eval(outline)
#     selected_subsections = parse_outline_with_subsections(outline)
    
#     full_survey_content = process_outline_with_empty_sections_citations(
#         parsed_outline, 
#         selected_subsections, 
#         context_list, 
#         client, 
#         citation_data_list
#     )
    # # Generate introduction and replace the existing one
    # generated_introduction = generate_introduction_alternate(title, full_survey_content, client)
    # introduction_pattern = r"(# 2 Introduction\n)(.*?)(\n# 3 )"
    # full_survey_content = re.sub(introduction_pattern, rf"\1{generated_introduction}\n\3", full_survey_content, flags=re.DOTALL)

    # return full_survey_content

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
