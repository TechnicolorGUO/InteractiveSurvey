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

from .asg_retriever import Retriever

def getQwenClient(): 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_base = os.environ.get("OPENAI_API_BASE")
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

def generate_introduction_alternate(context, client):

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

def generate_survey_paper_new(outline, context_list, client):
    parsed_outline = ast.literal_eval(outline)
    selected_subsections = parse_outline_with_subsections(outline)

    full_survey_content = process_outline_with_empty_sections_new(parsed_outline, selected_subsections, context_list, client)

    generated_introduction = generate_introduction_alternate(full_survey_content, client)
    
    introduction_pattern = r"(# 2 Introduction\n)(.*?)(\n# 3 )"
    full_survey_content = re.sub(introduction_pattern, rf"\1{generated_introduction}\n\3", full_survey_content, flags=re.DOTALL)

    return full_survey_content
context_list = [
    "Context for Predictive Modeling on Imbalance Data",
    "Context for Class Imbalance as a Major Challenge",
    "Context for Survey Scope and Organization",
    "Context for Dealing with Class Imbalance",
    "Context for Clustering Machine Learning Methods",
    "Context for Microtubule Dynamics and Behavior",
    "Context for Other Methods",
    "Context for Conclusion",
]

# 新的调用方式
# generated_survey_paper = generate_survey_paper_new(outline1, context_list, client)


context = '''
Many paradigms have been proposed to asses informativeness of data samples for active learning. One of the popular approaches is selecting the most uncertain data sample, i.e the data sample in which current classifier is least confident. Some other approaches are selecting the sample which yields a model with minimum risk or the data sample which yields fastest convergence in gradient based methods.//
An active under-sampling approach is presented in this paper to change the data distribution of training datasets, and improve the classification accuracy of minority classes while maintaining overall classification performance.//
In this paper, we propose an uncertainty-based active learning algorithm which requires only samples of one class and a set of unlabeled data in order to operate.//
The principal contribution of our work is twofold: First, we use Bayes’ rule and density estimation to avoid the need to have a model of all classes for computing the uncertainty measure.//
This technique reduces the number of input parameters of the problem. At the rest of this paper, we first review recent related works in the fields of active learning and active one-class learning (section II).//
The classifier predicts that all the samples are non-fraud, it will have a quite high accuracy. However, for problems like fraud detection, minority class classification accuracy is more critical.//
The algorithm used and the features selected are always the key points at design time, and many experiments are needed to select the final algorithm and the best suited feature set.//
Active learning works by selecting among unlabeled data, the most informative data sample. The informativeness of a sample is the amount of accuracy gain achieved after adding it to the training set.//
Some other approaches are selecting the sample which yields a model with minimum risk or the data sample which yields fastest convergence in gradient based methods.//
In this paper, we propose a novel approach reducing each within group error, BABoost, that is a variant of AdaBoost.//
Simulations on different unbalanced distribution data and experiments performed on several real datasets show that the new method is able to achieve a lower within group error.//
Active learning with early stopping can achieve a faster and scalable solution without sacrificing prediction performance.//
We also propose an efficient Support Vector Machine (SVM) active learning strategy which queries a small pool of data at each iterative step instead of querying the entire dataset.//
The second part consists of applying a treatment method and inducing a classifier for each class distribution.//
This time we measured the percentage of the performance loss that was recovered by the treatment method.//
We used two well-known over-sampling methods, random over-sampling and SMOTE.//
We tested our proposed technique on a sample of three representative functional genomic problems: splice site, protein subcellular localization and phosphorylation site prediction problems.//
Among the possible PTMs, phosphorylation is the most studied and perhaps the most important.//
The second part consists of applying a treatment method and inducing a classifier for each class distribution.//
We show that Active Learning (AL) strategy can be a more efficient alternative to resampling methods to form a balanced training set for the learner in early stages of the learning.//
'''

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
    response = generateResponse(client, formatted_prompt)

    # # 解析生成的内容
    # content_start = f"Survey Paper Content for \"{section_title}\":"
    # start_index = response.rfind(content_start)
    
    # if start_index != -1:
    #     # 去除提示和引导部分，保留生成的内容
    #     answer = response[start_index + len(content_start):].strip()
    # else:
    #     answer = context  # 如果解析失败，返回原始 context

    # return answer
    return response.strip()


# def generate_survey_paper(outline, context, client):
#     '''
#     Original version
#     '''
#     parsed_outline = ast.literal_eval(outline)
#     selected_subsections = parse_outline_with_subsections(outline)

#     # 生成并处理完整的 survey paper 内容
#     full_survey_content = process_outline_with_empty_sections(parsed_outline, selected_subsections, context, client)
#     return full_survey_content

def generate_survey_paper(outline, context, client):
    parsed_outline = ast.literal_eval(outline)
    selected_subsections = parse_outline_with_subsections(outline)

    # 生成并处理完整的 survey paper 内容
    full_survey_content = process_outline_with_empty_sections(parsed_outline, selected_subsections, context, client)

    # 替换生成的引言部分
    generated_introduction = generate_introduction_alternate(full_survey_content, client)
    
    # 正则表达式匹配引言内容，但保持标题完整
    introduction_pattern = r"(# 2 Introduction\n)(.*?)(\n# 3 )"  # 匹配从 # 2 Introduction 到下一个顶级 section 之间的内容
    full_survey_content = re.sub(introduction_pattern, rf"\1{generated_introduction}\n\3", full_survey_content, flags=re.DOTALL)

    return full_survey_content

def query_embedding_for_title(collection_name: str, title: str, n_results: int = 1):
    final_context = ""
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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

    subsections = parse_outline_with_subsections(outline)
    for level, title in subsections:
        if(title.startswith("3")):
            cluster_idx = 0
        elif(title.startswith("4")):
            cluster_idx = 1
        elif(title.startswith("5")):
            cluster_idx = 2
        print(f"Generating context for section: {title}")
        context_temp = ""
        for i in range(len(collection_list[cluster_idx])):
            context = query_embedding_for_title(collection_list[cluster_idx][i], title)
            context_temp += context
            context_temp += "\n"
        context_list.append(context_temp)
    print(f"Context list generated with length {len(context_list)}.")
    return context_list




outline1 = """
[
    [1, '1 Abstract'], 
    [1, '2 Introduction'],
        [2, '2.1 Predictive Modeling on Imbalance Data: Problem Overview'], 
        [2, '2.2 Class Imbalance: A Major Challenge in Predictive Modeling'], 
        [2, '2.3 Survey Scope and Organization'], 
            [3, '2.3.1 Dealing with Class Imbalance'], 
            [3, '2.3.2 Clustering Machine Learning Methods'], 
            [3, '2.3.3 Microtubule Dynamics and Behavior'], 
            [3, '2.3.4 Other Methods'], 
        [2, '2.4 Conclusion'], 
    [1, '3 Dealing with Class Imbalance'], 
        [2, '3.1 An Experimental Design to Evaluate Class Imbalance Treatment Methods'], 
        [2, '3.2 An Improved SMOTE Imbalanced Data Classification Method Based on Support Degree'], 
    [1, '4 Clustering Machine Learning Methods'], 
        [2, '4.1 An unsupervised learning approach to resolving the data imbalanced issue in supervised learning problems in functional genomics'], 
    [1, '5 Microtubule Dynamics and Behavior'], 
        [2, '5.1 Active Learning for Class Imbalance Problem'], 
    [1, '6 Conclusion'], 
    [1, '7 Future Research Directions'], 
    [1, '8 References']
]
"""
# client = getQwenClient()

# generated_survey_paper = generate_survey_paper(outline1, context, client)
# print("Generated Survey Paper:\n", generated_survey_paper)

# generated_introduction = generate_introduction(generated_survey_paper, client)
# print("\nGenerated Introduction:\n", generated_introduction)