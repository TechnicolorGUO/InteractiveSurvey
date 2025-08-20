import transformers
import torch
import os
import json
import re
import ast
from survey_generator_api import *
from asg_abstract import AbstractGenerator
from asg_conclusion import ConclusionGenerator
from asg_retriever import *
import pandas as df
from references import generate_references


class OutlineGenerator():
    def __init__(self, pipeline, df, cluster_names, mode='desp'):
        self.pipeline = pipeline
        # self.pipeline.model.load_adapter("technicolor/llama3.1_8b_outline_generation")
        self.pipeline.model.set_adapter("outline")
        self.df = df
        self.cluster = [{'label': i, 'name': cluster_names[i]} for i in range(len(cluster_names))]
        self._add_cluster_info()
        self.mode = mode

    def __init__(self, df, cluster_names, mode='desp'): #Without local llms
        self.df = df
        self.cluster = [{'label': i, 'name': cluster_names[i]} for i in range(len(cluster_names))]
        self._add_cluster_info()
        self.mode = mode

    def _add_cluster_info(self):
        label_to_info = {label: self.df[self.df['label'] == label] for label in range(len(self.cluster))}
        for cluster in self.cluster:
            cluster['info'] = label_to_info[cluster['label']]

    def get_cluster_info(self):
        return self.cluster

    def generate_claims(self):
        result = []
        if self.mode == 'desp':
            for i in range(len(self.cluster)):
                cluster = self.cluster[i]
                claims = ''
                for j in range(len(cluster['info'])):
                    claims = cluster['info'].iloc[j]['retrieval_result'] + '\n' + claims
                    # claims = cluster['info'].iloc[j]['ref_title'] + '\n' + claims
                result.append(claims)
        else:
            for i in range(len(self.cluster)):
                cluster = self.cluster[i]
                claims = ''
                data = cluster['info']
                for j in range(len(data)):
                    entry = data.iloc[j]
                    title = entry['title']
                    abstract = entry['abstract']
                    prompt = f'''
                        Title:
                        {title}
                        Abstract:
                        {abstract}
                        Task:
                        Conclude new findings and null findings from the abstract in one sentence in the atomic format. Do not separate
                        new findings and null findings. The finding must be relevant to the title. Do not include any other information.
                        Definition:
                        A scientific claim is an atomic verifiable statement expressing a finding about one aspect of a scientific entity or
                        process, which can be verified from a single source.'''
                    
                    messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                    ]
                    
                    outputs = self.pipeline(
                        messages,
                        max_new_tokens=256,
                    )
                    claim = outputs[0]["generated_text"][-1]['content']
                    # print(claim)
                    # print('+++++++++++++++++++++++++++++++++')
                    claims = claims + '\n' + claim
                result.append(claims)
        return result
    
    
    def generate_claims_qwen(self):
        """
        Generate claims for each cluster using Qwen API.

        Returns:
            list: A list of strings, where each string contains the claims generated
                for a cluster.
        """
        result = []
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        for i in range(len(self.cluster)):
            cluster = self.cluster[i]
            claims = ''
            data = cluster['info']

            for j in range(len(data)):
                entry = data.iloc[j]
                title = entry['title']
                abstract = entry['abstract']
                
                # Construct the prompt for Qwen
                prompt = f'''
                    Title:
                    {title}
                    Abstract:
                    {abstract}
                    Task:
                    Conclude new findings and null findings from the abstract in one sentence in the atomic format. Do not separate
                    new findings and null findings. The finding must be relevant to the title. Do not include any other information.
                    Definition:
                    A scientific claim is an atomic verifiable statement expressing a finding about one aspect of a scientific entity or
                    process, which can be verified from a single source.
                '''
                
                # Define the input for Qwen
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]

                try:
                    # Call Qwen API
                    chat_response = client.chat.completions.create(
                        model=os.environ.get("MODEL"),
                        max_tokens=32768,
                        temperature=0.5,
                        messages=messages,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    
                    # Extract the generated claim from Qwen's response
                    claim = ""
                    for chunk in chat_response:
                        if "content" in chunk.choices[0].delta:
                            claim += chunk.choices[0].delta.content
                    
                    # Clean and append the claim
                    claims = claims + '\n' + claim.strip()
                    # print("Generated claim:", claim)
                    # print("+++++++++++++++++++++++++++++++++")
                
                except Exception as e:
                    print(f"Error generating claim for entry {j} in cluster {i}: {e}")
                    continue

            result.append(claims)

        return result

    def generate_outline(self, survey_title):
        claims = self.generate_claims()
        cluster_with_claims = ""
        for i in range(len(self.cluster)):
            cluster = self.cluster[i]
            cluster_with_claims = cluster_with_claims + f'Cluster {i}: {cluster["name"]}\n' + "Descriptions for entities in this cluster: \n" + claims[i] + '\n\n'
        # system_prompt = f'''
        #     You are a helpful assistant who is helping a researcher to generate an outline for a survey paper.
        #     The references used by this survey paper have been clustered into different categories.
        #     The researcher will provides you with the title of the survey paper
        #     together with the cluster names and the descriptions for entities in each cluster.
        #     '''
        system_prompt = f'''Generate the outline of the survey paper following the format of the example : [[1, '1 Introduction'], [1, '2 Perturbations of (co)differentials'], [2, '2.1 Derivations of the tensor algebra'], [more sections...]].\
        The first element in the sub-list refers to the hierachy of the section name (from 1 to 3). Sections like Introduction and Conclusion should have the highest level (1)\
        The second element in the sub-list refers to the section name.
        '''

        example_json = {"title":"A Survey of Huebschmann and Stasheff's Paper: Formal Solution of the Master Equation via HPT and Deformation Theory","outline":[{"title":"1 Introduction","outline":[]},{"title":"2 Perturbations of (co)differentials","outline":[{"title":"2.1 Derivations of the tensor algebra","outline":[]},{"title":"2.2 Coderivations of the tensor coalgebra","outline":[]},{"title":"2.3 Coderivations of the symmetric coalgebra","outline":[]},{"title":"2.4 DGLA\u2019s and perturbations of the codifferential","outline":[]},{"title":"2.5 Strongly homotopy Lie algebras","outline":[]},{"title":"2.6 The Hochschild chain complex and DGA\u2019s","outline":[]},{"title":"2.7 Strongly homotopy associative algebras","outline":[]}]},{"title":"3 Master equation","outline":[]},{"title":"4 Twisting cochain","outline":[{"title":"4.1 Differential on Hom","outline":[]},{"title":"4.2 Cup product and cup bracket","outline":[]},{"title":"4.3 Twisting cochain","outline":[]}]},{"title":"5 Homological perturbation theory (HPT)","outline":[{"title":"5.1 Contraction","outline":[]},{"title":"5.2 The first main theorem.","outline":[]}]},{"title":"6 Corollaries and the second main theorem","outline":[{"title":"6.1 Other corollaries of Theorem\u00a01.","outline":[]},{"title":"6.2 The second main theorem","outline":[]}]},{"title":"7 Differential Gerstenhaber and BV algebras","outline":[{"title":"7.1 Differential Gerstenhaber algebras","outline":[]},{"title":"7.2 Differential BV algebras","outline":[]},{"title":"7.3 Formality","outline":[{"title":"7.3.1 Formality of differential graded P\ud835\udc43Pitalic_P-algebras","outline":[]},{"title":"7.3.2 Examples","outline":[]}]},{"title":"7.4 Differential BV algebras and formality","outline":[]}]},{"title":"8 Deformation theory","outline":[]},{"title":"References","outline":[]}]}
        # user_prompt = {"survey_title":survey_title, "claims":cluster_with_claims}
        user_prompt = f'''Generate the outline of the survey paper given the title:{survey_title}, and three lists of sentences describing each cluster of the references used by this survey:{cluster_with_claims}'''

        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content":"[[1, '1 Abstract'], [1, '2 Introduction'], "}
            ] 

        outputs = self.pipeline(
            messages,
            max_new_tokens=9192,
        )
        result = outputs[0]["generated_text"][-1]['content']

        self.pipeline.model.disable_adapters()

        return messages, result

    def generate_outline_qwen(self, survey_title, cluster_num = 3):
        claims = self.generate_claims()
        cluster_with_claims = ""
        cluster_names = []
        for i in range(cluster_num):  # 改为 cluster_num
            cluster = self.cluster[i]
            cluster_with_claims += f'Cluster {i}: {cluster["name"]}\nDescriptions for reference papers in this cluster:\n{claims[i]}\n\n'
            cluster_names.append(cluster["name"])
        # system_prompt = f'''
        #     You are a helpful assistant who is helping a researcher to generate an outline for a survey paper.
        #     The references used by this survey paper have been clustered into different categories.
        #     The researcher will provides you with the title of the survey paper
        #     together with the cluster names and the descriptions for entities in each cluster.
        #     '''
        system_prompt = f'''Finish the outline of the survey paper following the format of the example : [[1, '1 Introduction'], [1, '2 Perturbations of (co)differentials'], [2, '2.1 Derivations of the tensor algebra'], [3, '2.2.1 ...']......].\
        The first element in the sub-list refers to the hierachy of the section name (from 1 to 3). Sections like Introduction and Conclusion should have the highest level (1)\
        The second element in the sub-list refers to the section name.
        You are required to finish the second and third level subsections name under [1, '3 <Cluster 0's name>'], [1, '4 <Cluster 1's name>'] and [1, '5 <Cluster 2's name>']
        You must not generate third level susections over *3* for each second level subsection, for example, [3, '3.1.4 xxx'], [3, '3.1.5 xxx'] are not allowed. 
        *Try to conclude the main findings of each cluster in the second and third level subsections, use highly abstract terms and phrases to describe*
        *Do not include colons, e.g. AutoSurvey: Large Language Models Can Automatically Write Surveys should be written in Large Language Models in Writing Surveys*
        '''
        # user_prompt = {"survey_title":survey_title, "claims":cluster_with_claims}
        cluster_sections = "\n".join([f"[1, '{i+3} {cluster_names[i]}'], [level 2 and 3 sections to finish...]" for i in range(cluster_num)])

        user_prompt = f'''Finish the outline of the survey paper given the title: {survey_title}, and lists of sentences describing each cluster of the references used by this survey:\n{cluster_with_claims}
        The first level sections' hierarchy is given: [[1, '1 Abstract'], [1, '2 Introduction'], {cluster_sections}, [1, '{cluster_num+3} Future Directions'], [1, '{cluster_num+4} Conclusion']].
        You are required to finish the second and third level subsections under each cluster section with [2, 'a.b xxx'] and [3, 'a.b.c xxx'].
        You must not generate third level susections over *3* for each second level subsection, for example, [3, '3.1.4 xxx'], [3, '3.1.5 xxx'] are not allowed.
        *Try to conclude the main findings of each cluster in the second and third level subsections, use highly abstract terms and phrases to describe*
        *Do not include colons, e.g. AutoSurvey: Large Language Models Can Automatically Write Surveys should be written in Large Language Models in Writing Surveys*
        '''

        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt},
            ]
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")
        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key = openai_api_key,
            base_url = openai_api_base,
        )
        chat_response = client.chat.completions.create(
            model=os.environ.get("MODEL"),
            max_tokens=32768,
            temperature=0.5,
            stop="<|im_end|>",
            stream=True,
            messages= messages,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        # Stream the response to console
        text = ""
        for chunk in chat_response:
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        # print('The response is :', text)
        pattern = r'\[(.*)\]'
        match = re.search(pattern, text, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符
        text = match.group(1)
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return messages, clean_text

def parseOutline(survey_id, info_path = './src/static/data/txt'):
    file_path = f'{info_path}/{survey_id}/outline.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

    response = data.get('outline', '')
    if not response:
        print("No outline content found in JSON.")
        return []

    # 提取文本中第一个 '[' 与最后一个 ']' 之间的内容
    def extract_first_last(text):
        first_match = re.search(r'\[', text)
        last_match = re.search(r'\](?!.*\])', text)  # 使用负向前瞻查找最后一个 ']'
        if first_match and last_match:
            return '[' + text[first_match.start() + 1:last_match.start()] + ']'
        return None

    response_extracted = extract_first_last(response)
    if not response_extracted:
        print("Failed to extract a valid list string from the outline content.")
        return []

    # 检查提取结果是否为“列表的列表”格式（应该以 "[[" 开头）
    fixed_str = response_extracted.strip()
    if not fixed_str.startswith("[["):
        # 如果不是，则去掉原有的首尾括号，再重新包装：[[ ... ]]
        # 注意：这种方式假定内部结构是以逗号分隔的多个列表，而不是单个列表。
        fixed_str = "[[" + fixed_str[1:-1] + "]]"
        # 或者根据你的实际情况，也可简单包装外层括号：
        # fixed_str = "[" + fixed_str + "]"

    try:
        outline_list = ast.literal_eval(fixed_str)
    except Exception as e:
        print(f"Error converting extracted outline to a list.\nExtracted text: {fixed_str}\nError: {e}")
        return []

    # 如果结果不是列表，则转换成列表
    if not isinstance(outline_list, list):
        outline_list = list(outline_list)

    # 如果解析结果不是列表的列表，而是单个列表（例如 [a, b, c]），则将其包装成一个列表
    if outline_list and not all(isinstance(item, list) for item in outline_list):
        outline_list = [outline_list]

    result = []
    for item in outline_list:
        result.append(item)
    return result


def generateOutlineHTML_qwen(survey_id):
    outline_list = parseOutline(survey_id)
    html = '''
    <div class="container-fluid w-50 d-flex flex-column justify-content-center align-items-center">

        <style>
            /* 不同层级的样式 */
            .level-1 {
                font-size: 20px;
                font-weight: bold;
                position: relative;
                padding-right: 40px; /* 为箭头留出空间 */
            }
            .level-2 {
                font-size: 18px;
                padding-left: 40px;
            }
            .level-3 {
                font-size: 16px;
                padding-left: 80px;
            }
            .list-group-item {
                border: none;
            }
            
            /* 自定义卡片样式 */
            .custom-card {
                background-color: #fff;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
                width: 100%;
                max-width: 800px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 
                            0 6px 20px rgba(0, 0, 0, 0.1);
            }

            /* 自定义卡片主体样式 */
            .custom-card-body {
                padding: 20px;
            }

            /* 折叠图标样式 */
            .collapse-icon {
                background: none;
                border: none;
                padding: 0;
                position: absolute;
                right: 10px;
                top: 50%;
                transform: translateY(-50%) rotate(0deg);
                cursor: pointer;
                font-size: 16px;
                /* 旋转过渡效果 */
                transition: transform 0.2s;
            }
            /* 去除按钮聚焦时的轮廓 */
            .collapse-icon:focus {
                outline: none;
            }
            /* 当折叠展开时旋转图标 */
            .collapsed .collapse-icon {
                transform: translateY(-50%) rotate(0deg);
            }
            .in .collapse-icon {
                transform: translateY(-50%) rotate(90deg);
            }
        </style>

        <div class="custom-card">
            <div class="custom-card-body" id="display-outline">
                <ul class="list-group list-group-flush">
    '''

    # 添加默认的一级标题内容
    default_items = []

    # 将默认项与解析出的纲要列表合并
    combined_list = default_items + outline_list

    # 构建树形结构，以便检测一级标题是否有子标题
    def build_outline_tree(outline_list):
        sections = []
        stack = []
        for level, content in outline_list:
            level = int(level)
            node = {'level': level, 'content': content, 'subitems': []}
            if level == 1:
                sections.append(node)
                stack = [node]
            elif level == 2:
                if stack:
                    parent = stack[-1]
                    parent['subitems'].append(node)
                    # stack.append(node)
                else:
                    sections.append(node)
            elif level == 3:
                if stack:
                    parent = stack[-1]
                    parent['subitems'].append(node)
                else:
                    sections.append(node)
        return sections

    sections = build_outline_tree(combined_list)

    # 生成 HTML
    def generate_html_from_sections(sections):
        html = ''
        section_index = 1  # 用于生成唯一的 ID

        def generate_node_html(node):
            nonlocal section_index
            level = node['level']
            content = node['content']
            has_subitems = len(node['subitems']) > 0
            if level == 1:
                # 一级标题
                if has_subitems:
                    # 如果有子标题，添加下拉图标和可折叠功能
                    section_id = f"outline_collapseSection{section_index}"
                    section_index += 1
                    node_html = f'''
                        <li class="list-group-item level-1">
                            {content}
                            <a class="collapsed" data-toggle="collapse" data-target="#{section_id}" aria-expanded="true" aria-controls="{section_id}">
                                &#9654; <!-- 右箭头表示折叠状态 -->
                            </a>
                            <ul class="list-group collapse in" id="{section_id}">
                    '''
                    for subitem in node['subitems']:
                        node_html += generate_node_html(subitem)
                    node_html += '''
                            </ul>
                        </li>
                    '''
                else:
                    # 如果没有子标题，不显示下拉图标
                    node_html = f'''
                        <li class="list-group-item level-1">
                            {content}
                        </li>
                    '''
            elif level == 2:
                    node_html = f'<li class="list-group-item level-2">{content}</li>'
            elif level == 3:
                # 三级标题直接显示，已经在二级标题中处理
                node_html = f'<li class="list-group-item level-3">{content}</li>'
            return node_html

        for section in sections:
            html += generate_node_html(section)

        return html
    
    def generate_list_html(combined_list, editable=True):
        html = '<ul class="list-group list-group-flush">\n'  # 开始 <ul>
        for level, content in combined_list:
            # 根据层级添加对应的 class
            if level == 1:  # Level 1 的输入框需要禁用
                if editable:
                    html += f'<li class="list-group-item level-1"><input type="text" class="form-control" value="{content}" disabled></li>\n'
                else:
                    html += f'<li class="list-group-item level-1">{content}</li>\n'
            elif level == 2:
                if editable:
                    html += f'<li class="list-group-item level-2" style="padding-left: 20px;"><input type="text" class="form-control" value="{content}"></li>\n'
                else:
                    html += f'<li class="list-group-item level-2" style="padding-left: 20px;">{content}</li>\n'
            elif level == 3:
                if editable:
                    html += f'<li class="list-group-item level-3" style="padding-left: 40px;"><input type="text" class="form-control" value="{content}"></li>\n'
                else:
                    html += f'<li class="list-group-item level-3" style="padding-left: 40px;">{content}</li>\n'
        html += '</ul>'  # 结束 </ul>
        return html

    # 生成列表 HTML
    list_html = generate_list_html(combined_list)
    html += generate_html_from_sections(sections)

    html += f'''
                </ul>
            </div>
            <div class="custom-card-body" style="display: none" id="edit-outline">
                {list_html}
            </div>
            <button type="button" class="btn btn-secondary btn-lg" id="edit-btn" onclick="editOutline()"><i class="bi bi-pen"></i></button>
            <button type="button" class="btn btn-success btn-lg" id="confirm-btn" style="display: none;" onclick="confirmOutline()"><i class="bi bi-check"></i></button>
        </div>
        <!-- 添加 Bootstrap v3.3.0 的 JavaScript 来处理折叠功能 -->
        <script>
        $(document).ready(function(){{
            $('.collapsed').click(function(){{
                $(this).toggleClass('collapsed');
            }});
        }});
        </script>

    </div>
    '''

    html+='''
        <script>
        // 切换到编辑模式
        function editOutline() {
            document.getElementById("display-outline").style.display = "none"; // 隐藏不可编辑部分
            document.getElementById("edit-outline").style.display = "block";  // 显示可编辑部分

            // 显示 "Confirm" 按钮，隐藏 "Edit" 按钮
            document.getElementById("edit-btn").style.display = "none";
            document.getElementById("confirm-btn").style.display = "inline-block";
        }

        // 确认编辑并提交数据
function confirmOutline() {
    const outlineData = []; // 用于存储提交到后端的数据

    // 遍历所有的可编辑输入框
    document.querySelectorAll("#edit-outline .list-group-item").forEach((item) => {
        const level = item.classList.contains("level-1") ? 1 :
                      item.classList.contains("level-2") ? 2 : 3; // 获取层级
        const content = item.querySelector("input").value.trim(); // 获取编辑框的值

        // 将数据转换为数组格式 [level, content]
        outlineData.push([level, content]); 
    });

    console.log("Submitting to backend:", outlineData); // 打印提交数据以供调试

    // 使用 AJAX 提交数据到后端
    const csrftoken = getCookie("csrftoken"); // 获取 CSRF token
    fetch("/save_outline/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrftoken, // Django 的 CSRF 令牌
        },
        body: JSON.stringify({ outline: outlineData }) // 将数据转换为 JSON 字符串
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.status === "success") {
                $('#sections_').html(data.html);
                alert("Outline updated successfully!");
            } else {
                alert("Error updating outline: " + data.message);
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("Error updating outline. Please check the console for details.");
        });
}
        </script>
        '''
    return html

def insert_section(content, section_header, section_content):
    """
    在 content 中找到以 section_header 开头的行，并在其后插入 section_content
    section_header: 标题名称，例如 "Abstract" 或 "Conclusion"
    section_content: 要插入的内容（字符串）
    """
    # 修改正则表达式，使得数字后的点是可选的
    pattern = re.compile(
        r'(^#\s+\d+\.?\s+' + re.escape(section_header) + r'\s*$)',
        re.MULTILINE | re.IGNORECASE
    )
    replacement = r'\1\n\n' + section_content + '\n'
    new_content, count = pattern.subn(replacement, content)
    if count == 0:
        print(f"警告: 未找到标题 '{section_header}'。无法插入内容。")
    return new_content

def generateOutlineHTML(survey_id):
    outline_list = parseOutline(survey_id)
    html = '''
    <div class="container-fluid w-50 d-flex flex-column justify-content-center align-items-center">

        <style>
            /* 不同层级的样式 */
            .level-1 {
                font-size: 20px;
                font-weight: bold;
                position: relative;
                padding-right: 40px; /* 为箭头留出空间 */
            }
            .level-2 {
                font-size: 18px;
                padding-left: 40px;
            }
            .level-3 {
                font-size: 16px;
                padding-left: 80px;
            }
            .list-group-item {
                border: none;
            }
            
            /* 自定义卡片样式 */
            .custom-card {
                background-color: #fff;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
                width: 100%;
                max-width: 800px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 
                            0 6px 20px rgba(0, 0, 0, 0.1);
            }

            /* 自定义卡片主体样式 */
            .custom-card-body {
                padding: 20px;
            }

            /* 折叠图标样式 */
            .collapse-icon {
                background: none;
                border: none;
                padding: 0;
                position: absolute;
                right: 10px;
                top: 50%;
                transform: translateY(-50%) rotate(0deg);
                cursor: pointer;
                font-size: 16px;
                /* 旋转过渡效果 */
                transition: transform 0.2s;
            }
            /* 去除按钮聚焦时的轮廓 */
            .collapse-icon:focus {
                outline: none;
            }
            /* 当折叠展开时旋转图标 */
            .collapsed .collapse-icon {
                transform: translateY(-50%) rotate(0deg);
            }
            .in .collapse-icon {
                transform: translateY(-50%) rotate(90deg);
            }
        </style>

        <div class="custom-card">
            <div class="custom-card-body">
                <ul class="list-group list-group-flush">
    '''

    # 添加默认的一级标题内容
    default_items = [[1, '1 Abstract'], [1, '2 Introduction']]

    # 将默认项与解析出的纲要列表合并
    combined_list = default_items + outline_list

    # 构建树形结构，以便检测一级标题是否有子标题
    def build_outline_tree(outline_list):
        sections = []
        stack = []
        for level, content in outline_list:
            level = int(level)
            node = {'level': level, 'content': content, 'subitems': []}
            if level == 1:
                sections.append(node)
                stack = [node]
            elif level == 2:
                if stack:
                    parent = stack[-1]
                    parent['subitems'].append(node)
                    # stack.append(node)
                else:
                    sections.append(node)
            elif level == 3:
                if stack:
                    parent = stack[-1]
                    parent['subitems'].append(node)
                else:
                    sections.append(node)
        return sections

    sections = build_outline_tree(combined_list)

    # 生成 HTML
    def generate_html_from_sections(sections):
        html = ''
        section_index = 1  # 用于生成唯一的 ID

        def generate_node_html(node):
            nonlocal section_index
            level = node['level']
            content = node['content']
            has_subitems = len(node['subitems']) > 0
            if level == 1:
                # 一级标题
                if has_subitems:
                    # 如果有子标题，添加下拉图标和可折叠功能
                    section_id = f"outline_collapseSection{section_index}"
                    section_index += 1
                    node_html = f'''
                        <li class="list-group-item level-1">
                            {content}
                            <a class="collapsed" data-toggle="collapse" data-target="#{section_id}" aria-expanded="true" aria-controls="{section_id}">
                                &#9654; <!-- 右箭头表示折叠状态 -->
                            </a>
                            <ul class="list-group collapse in" id="{section_id}">
                    '''
                    for subitem in node['subitems']:
                        node_html += generate_node_html(subitem)
                    node_html += '''
                            </ul>
                        </li>
                    '''
                else:
                    # 如果没有子标题，不显示下拉图标
                    node_html = f'''
                        <li class="list-group-item level-1">
                            {content}
                        </li>
                    '''
            elif level == 2:
                    node_html = f'<li class="list-group-item level-2">{content}</li>'
            elif level == 3:
                # 三级标题直接显示，已经在二级标题中处理
                node_html = f'<li class="list-group-item level-3">{content}</li>'
            return node_html

        for section in sections:
            html += generate_node_html(section)

        return html

    html += generate_html_from_sections(sections)

    html += '''
                </ul>
            </div>
        </div>
        <!-- 添加 Bootstrap v3.3.0 的 JavaScript 来处理折叠功能 -->
        <script>
        $(document).ready(function(){
            // 切换箭头方向
            $('.collapsed').click(function(){
                $(this).toggleClass('collapsed');
            });
        });
        </script>
    </div>
    '''
    return html

def insert_section(content, section_header, section_content):
    """
    在 content 中找到以 section_header 开头的行，并在其后插入 section_content
    section_header: 标题名称，例如 "Abstract" 或 "Conclusion"
    section_content: 要插入的内容（字符串）
    """
    # 修改正则表达式，使得数字后的点是可选的
    pattern = re.compile(
        r'(^#\s+\d+\.?\s+' + re.escape(section_header) + r'\s*$)',
        re.MULTILINE | re.IGNORECASE
    )
    replacement = r'\1\n\n' + section_content + '\n'
    new_content, count = pattern.subn(replacement, content)
    if count == 0:
        print(f"警告: 未找到标题 '{section_header}'。无法插入内容。")
    return new_content

def generateSurvey(survey_id, title, collection_list, pipeline):
    outline = parseOutline(survey_id)
    default_items = [[1, '1 Abstract'], [1, '2 Introduction'], [1, '3 Overview']]
    outline = str(default_items + outline)
    
    client = getQwenClient()

    context_list = generate_context_list(outline, collection_list)

    temp = {
        "survey_id": survey_id,
        "outline": str(default_items), 
        "survey_title": title,
        "context": context_list, 
        "abstract": "",
        "introduction": "",
        "content": "",
        "conclusion": "",
        "references": ""
    }

    generated_survey_paper = generate_survey_paper_new(outline, context_list, client)

    generated_introduction = generate_introduction(generated_survey_paper, client)
    # print("\nGenerated Introduction:\n", generated_introduction)

    abs_generator = AbstractGenerator(pipeline)
    abstract = abs_generator.generate(title, generated_introduction)
    con_generator = ConclusionGenerator(pipeline)
    conclusion = con_generator.generate(title, generated_introduction)

    abstract = abstract.replace("Abstract:", "")
    conclusion = conclusion.replace("Conclusion:", "")

    temp["abstract"] = abstract
    temp["introduction"] = generated_introduction
    temp["content"] = generated_survey_paper
    temp["conclusion"] = conclusion

    temp["content"] = insert_section(temp["content"], "Abstract", temp["abstract"])
    temp["content"] = insert_section(temp["content"], "Conclusion", temp["conclusion"])

    output_path = f'./src/static/data/txt/{survey_id}/generated_result.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(temp, f, ensure_ascii=False, indent=4)
    print(f"Survey has been saved to {output_path}.")

    return

def generate_future_directions_qwen(client, title, intro):
    system_prompt = f'''You are a helpful assistant that help to generate the future directions of the survey paper given the survey title and survey introduction.'''
    # user_prompt = {"survey_title":survey_title, "claims":cluster_with_claims}
    user_prompt = f'''Help me to generate the future directions of a survey paper given the title: *{title}*, and and the introduction:{intro} within 300 words.'''

    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content":"Future Directions:"}
        ] 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        max_tokens=32768,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages= messages,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    # Stream the response to console
    text = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    return text

def generateSurvey_qwen(survey_id, title, collection_list, pipeline):
    outline = str(parseOutline(survey_id))
    
    client = getQwenClient()

    context_list = generate_context_list(outline, collection_list)

    temp = {
        "survey_id": survey_id,
        "outline": outline, 
        "survey_title": title,
        "context": context_list, 
        "abstract": "",
        "introduction": "",
        "content": "",
        "future_directions":"",
        "conclusion": "",
        "references": ""
    }

    generated_survey_paper = generate_survey_paper_new(title, outline, context_list, client)
    # print("Generated Survey Paper:\n", generated_survey_paper)

    generated_introduction = generate_introduction(generated_survey_paper, client)
    # print("\nGenerated Introduction:\n", generated_introduction)
    abs_generator = AbstractGenerator(pipeline)
    abstract = abs_generator.generate(title, generated_introduction)
    con_generator = ConclusionGenerator(pipeline)
    # conclusion = con_generator.generate(title, generated_introduction)
    #New version: 12/03
    conclusion = generate_conclusion(generated_survey_paper, client)
    abstract = abstract.replace("Abstract:", "")
    conclusion = conclusion.replace("Conclusion:", "")
    # future_directions =  generate_future_directions_qwen(client, title, generated_introduction).replace("Future Directions:","")
    #New version: 12/03
    future_directions = generate_future_work(generated_survey_paper, client)
    # references = generate_references_dir('./src/static/data/txt/'+survey_id)
    temp["abstract"] = abstract
    temp["introduction"] = generated_introduction
    temp["content"] = generated_survey_paper
    temp["conclusion"] = conclusion
    temp["future_directions"] = future_directions
    # temp["references"] = "\n\n".join([f"{ref}" for i, ref in enumerate(references)])
    temp["content"] = insert_section(temp["content"], "Abstract", temp["abstract"])
    temp["content"] = insert_section(temp["content"], "Conclusion", temp["conclusion"])
    temp["content"] = insert_section(temp["content"], "Future Directions", temp["future_directions"])
    output_path = f'./src/static/data/txt/{survey_id}/generated_result.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(temp, f, ensure_ascii=False, indent=4)
    print(f"Survey has been saved to {output_path}.")
    return

# wza
def generateSurvey_qwen_new(survey_id, title, collection_list, pipeline, citation_data_list, txt_path = "./src/static/data/txt"):
    outline = str(parseOutline(survey_id,  info_path ='./info'))
    client = getQwenClient()
    context_list = generate_context_list(outline, collection_list)

    # print("!!!!!!!!")
    # print(context_list)
    # print("2025")  
    
    # 不再重复查询citation数据，而是直接使用传入的citation_data_list
    # citation_data_list来自get_survey_id传入的Global_citation_data

    temp = {
        "survey_id": survey_id,
        "outline": outline, 
        "survey_title": title,
        "context": context_list, 
        "abstract": "",
        "introduction": "",
        "content": "",
        "future_directions": "",
        "conclusion": "",
        "references": ""
    }

    # 调用generate_survey_paper_new时传入citation_data_list
    generated_survey_paper = generate_survey_paper_new(title, outline, context_list, client, citation_data_list)

    generated_introduction = generate_introduction_alternate(title, generated_survey_paper, client)
    # generated_introduction = introduction_with_citations(generated_introduction, citation_data_list)
    # print("\nGenerated Introduction:\n", generated_introduction)
    # abs_generator = AbstractGenerator(pipeline)
    # abstract = abs_generator.generate(title, generated_introduction)
    abstract = generate_abstract(generated_survey_paper, client)
    # con_generator = ConclusionGenerator(pipeline)
    # conclusion = con_generator.generate(title, generated_introduction)
    conclusion = generate_conclusion(generated_survey_paper, client)
    abstract = abstract.replace("Abstract:", "")
    conclusion = conclusion.replace("Conclusion:", "")
    # future_directions =  generate_future_directions_qwen(client, title, generated_introduction).replace("Future Directions:","")
    #New version: 12/03
    future_directions = generate_future_work(generated_survey_paper, client)
    # references = generate_references_dir('./src/static/data/txt/'+survey_id)
    temp["abstract"] = abstract
    temp["introduction"] = generated_introduction
    temp["content"] = generated_survey_paper
    temp["conclusion"] = conclusion
    temp["future_directions"] = future_directions
    # temp["references"] = "\n\n".join([f"{ref}" for i, ref in enumerate(references)])
    temp["content"] = insert_section(temp["content"], "Abstract", temp["abstract"])
    temp["content"] = insert_section(temp["content"], "Conclusion", temp["conclusion"])
    temp["content"] = insert_section(temp["content"], "Future Directions", temp["future_directions"])
    output_path = f'{txt_path}/{survey_id}/generated_result.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(temp, f, ensure_ascii=False, indent=4)
    print(f"Survey has been saved to {output_path}.")
    return


def generate_references_dir(dir):
    client = getQwenClient()
    papers_info = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            file_path = os.path.join(dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # print("The data is: ")
                # print(data)
                papers_info.append({
                    "file_path": file_path,
                    "title": data.get("title", "Unknown Title"),
                    "authors": data.get("authors", "Unknown Author")
                })
                print("The length of papers_info is: ", len(papers_info))
    references = generate_references(papers_info, client)
    return references

if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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

    collection_list = ['activelearningfrompositiveandunlabeleddata', ]

    Global_pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        token = os.getenv('HF_API_KEY'),
        device_map="auto",
    )
    Global_pipeline.model.load_adapter(peft_model_id = "technicolor/llama3.1_8b_outline_generation", adapter_name="outline")
    Global_pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_conclusion_generation", adapter_name="conclusion")
    Global_pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_abstract_generation", adapter_name="abstract")


    # generateOutlineHTML('test')
    generateSurvey("test", "Predictive modeling of imbalanced data", collection_list, Global_pipeline)
