import csv
import json
import os

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from asg_retriever import legal_pdf
from asg_loader import DocumentLoading
from asg_retriever import Retriever, query_embeddings_new_new
from asg_generator import generate_sentence_patterns, generate
from category_and_tsne import clustering
from langchain_text_splitters  import RecursiveCharacterTextSplitter
import time
import torch
import re
import transformers
from dotenv import load_dotenv
from asg_clustername import generate_cluster_name_new
from asg_outline import OutlineGenerator, generateSurvey_qwen_new
import os
from markdown_pdf import MarkdownPdf, Section  # Assuming you are using markdown_pdf
from typing import Any

def clean_str(input_str):
    input_str = str(input_str).strip().lower()
    if input_str == "none" or input_str == "nan" or len(input_str) == 0:
        return ""
    input_str = input_str.replace('\\n',' ').replace('\n',' ').replace('\r',' ').replace('——',' ').replace('——',' ').replace('__',' ').replace('__',' ').replace('........','.').replace('....','.').replace('....','.').replace('..','.').replace('..','.').replace('..','.').replace('. . . . . . . . ','. ').replace('. . . . ','. ').replace('. . . . ','. ').replace('. . ','. ').replace('. . ','. ')
    input_str = re.sub(r'\\u[0-9a-z]{4}', ' ', input_str).replace('  ',' ').replace('  ',' ')
    return input_str

def remove_invalid_citations(text, valid_collection_names):
    """
    只保留 [xxx\] 中的 xxx 属于 valid_collection_names 的引用，
    其余的引用标记一律删除。
    """
    pattern = r"\[(.*?)\\\]"  # 匹配形如 [xxx\] 的内容
    all_matches = re.findall(pattern, text)

    new_text = text
    for match in all_matches:
        cleaned_match = match.rstrip('\\')  # 去除末尾的 \
        if cleaned_match not in valid_collection_names:
            new_text = new_text.replace(f"[{match}\\]", "")
    return new_text
def normalize_citations_with_mapping(paper_text):
    # 使用正则表达式匹配所有引用标记（形如 [citation1]）
    citations = re.findall(r'\[.*?\]', paper_text)
    # 去重并保持顺序
    unique_citations = list(dict.fromkeys(citations))
    # 生成引用映射表，把原始引用标记映射为数字引用
    citation_mapping = {citation: f'[{i + 1}]' for i, citation in enumerate(unique_citations)}

    # 在文本中替换老引用为新引用
    normalized_text = paper_text
    for old_citation, new_citation in citation_mapping.items():
        normalized_text = normalized_text.replace(old_citation, new_citation)

    # 生成从数字到原始引用标记的反向映射
    # 用 rstrip('\\') 去掉末尾的反斜杠
    reverse_mapping = {
        i + 1: unique_citations[i].strip('[]').rstrip('\\')
        for i in range(len(unique_citations))
    }

    return normalized_text, reverse_mapping
def generate_references_section(citation_mapping, collection_pdf_mapping):
    
    references = ["# References"]  # 生成引用部分
    for num in sorted(citation_mapping.keys()):
        collection_name = citation_mapping[num]
        pdf_name = collection_pdf_mapping.get(collection_name, "Unknown PDF")
        if pdf_name.endswith(".pdf"):
            pdf_name = pdf_name[:-4]
        # 在每一行末尾添加两个空格以确保换行
        references.append(f"[{num}] {pdf_name}  ")

    return "\n".join(references)
def fix_citation_punctuation_md(text):
    """
    把类似于 'some text. \[1]' 或 'some text. \[2]' 调整为 'some text \[1].'
    仅针对已经变成 \[1], \[2] 之类数字引用的 Markdown 情况有效。
    如果还没有变成 \[数字]，则需先经过 normalize_citations_with_mapping。
    """
    # 正则表达式匹配点号后带有空格或无空格，紧接 \[数字] 的情况
    pattern = r'\.\s*(\\\[\d+\])'
    replacement = r' \1.'
    fixed_text = re.sub(pattern, replacement, text)
    return fixed_text
def finalize_survey_paper(paper_text, 
                          Global_collection_names, 
                          Global_file_names):

    # 1) 删除所有不想要的旧引用（包括 [数字]、[Sewon, 2021] 等）
    paper_text = remove_invalid_citations(paper_text, Global_collection_names)

    # 2) 规范化引用 => [1][2]...
    normalized_text, citation_mapping = normalize_citations_with_mapping(paper_text)
    
    # 3) 修复标点，比如 .[1] => [1].
    normalized_text = fix_citation_punctuation_md(normalized_text)

    # 4) 构造 {collection_name: pdf_file_name} 字典
    collection_pdf_mapping = dict(zip(Global_collection_names, Global_file_names))
    
    # 5) 生成 References
    references_section = generate_references_section(citation_mapping, collection_pdf_mapping)
    
    # 6) 合并正文和 References
    final_paper = normalized_text.strip() + "\n\n" + references_section
    return final_paper

class ASG_system:
    def __init__(self, root_path: str, survey_id:str, pdf_path: str, survey_title: str, cluster_standard: str) -> None:
        load_dotenv()
        self.pdf_path = pdf_path
        self.txt_path = root_path + "/txt"
        self.tsv_path = root_path + "/tsv"
        self.md_path = root_path + "/md"
        self.info_path = root_path + "/info"
        self.result_path = root_path + "/result"

        self.survey_id = survey_id
        self.survey_title = survey_title
        self.cluster_standard = cluster_standard

        self.collection_names = []
        self.file_names = []
        self.citation_data = []
        self.description_list = []
        self.ref_list = []
        self.cluster_names = []
        self.collection_names_clustered = []
        self.df_selected = ''


        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            token = os.getenv('HF_API_KEY'),
            device_map="auto",
        )
        self.pipeline.model.load_adapter(peft_model_id = "technicolor/llama3.1_8b_outline_generation", adapter_name="outline")
        self.pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_abstract_generation", adapter_name="abstract")
        self.pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_conclusion_generation", adapter_name="conclusion")

        os.makedirs(self.txt_path, exist_ok=True)
        os.makedirs(f'{self.txt_path}/{self.survey_id}', exist_ok=True)

        os.makedirs(self.tsv_path, exist_ok=True)

        os.makedirs(self.md_path, exist_ok=True)
        os.makedirs(f'{self.md_path}/{self.survey_id}', exist_ok=True)

        os.makedirs(self.info_path, exist_ok=True)
        os.makedirs(f'{self.info_path}/{self.survey_id}', exist_ok=True)

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(f'{self.result_path}/{self.survey_id}', exist_ok=True)

    def parsing_pdfs(self, mode="intro") -> None:
        pdf_files = os.listdir(self.pdf_path)
        loader = DocumentLoading()
        

        for pdf_file in pdf_files:

            pdf_file = os.path.join(self.pdf_path, pdf_file)

            split_start_time = time.time()

            base_name = os.path.splitext(os.path.basename(pdf_file))[0]
            target_dir = os.path.join(self.md_path, self.survey_id, base_name, "auto")
            md_dir = os.path.join(self.md_path, self.survey_id)

            loader.convert_pdf_to_md(pdf_file, md_dir)

            md_file_path = os.path.join(target_dir, f"{base_name}.md")
            print(md_file_path)
            print("*"*24)
            if not os.path.exists(md_file_path):
                raise FileNotFoundError(f"Markdown file {md_file_path} does not exist. Conversion might have failed.")

            if mode == "intro":
                doc = loader.process_md_file(md_file_path, self.survey_id, self.txt_path)
            elif mode == "full":
                doc = loader.process_md_file_full(md_file_path, self.survey_id,self.txt_path)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=30,
                length_function=len,
                is_separator_regex=False,
            )
            splitters = text_splitter.create_documents([doc])
            documents_list = [document.page_content for document in splitters]
            for i in range(len(documents_list)):
                documents_list[i] = documents_list[i].replace('\n', ' ')
            print(f"Splitting took {time.time() - split_start_time} seconds.")

            embed_start_time = time.time()

            doc_results = self.embedder.embed_documents(documents_list)
            if isinstance(doc_results, torch.Tensor):
                embeddings_list = doc_results.tolist()
            else:
                embeddings_list = doc_results
            print(f"Embedding took {time.time() - embed_start_time} seconds.")

            # Prepare metadata
            metadata_list = [{"doc_name": os.path.basename(pdf_file)} for i in range(len(documents_list))]

            title = os.path.splitext(os.path.basename(pdf_file))[0]
            

            title_new = title.strip()
            invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*','_']
            for char in invalid_chars:
                title_new = title_new.replace(char, ' ')
            print("============================")
            print(title_new)

            # New logic to create collection_name
            # filename = os.path.basename(file_path)
            collection_name = legal_pdf(title_new)

            retriever = Retriever()
            retriever.list_collections_chroma()
            retriever.create_collection_chroma(collection_name)
            retriever.add_documents_chroma(
                collection_name=collection_name,
                embeddings_list=embeddings_list,
                documents_list=documents_list,
                metadata_list=metadata_list
            )

            self.collection_names.append(collection_name)
            self.file_names.append(title_new)
        print(self.collection_names)
        print(self.file_names)

        json_files = os.listdir(os.path.join(self.txt_path, self.survey_id))
        ref_paper_num = len(json_files)
        print(f'The length of the json files is {ref_paper_num}')


        json_data_pd = pd.DataFrame()   
        for _ in json_files:
            file_path = os.path.join(self.txt_path, self.survey_id, _)

            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)

                # Extract necessary information
                title = data.get("title", "")
                abstract = data.get("abstract", "")
                authors = data.get("authors", "")
                introduction = data.get("introduction", "")
                new_data = {
                    "reference paper title": title,
                    "reference paper citation information (can be collected from Google scholar/DBLP)": authors,
                    "reference paper abstract (Please copy the text AND paste here)": abstract,
                    "reference paper introduction (Please copy the text AND paste here)": introduction,
                    "reference paper doi link (optional)": "",
                    "reference paper category label (optional)": ""
                }

                # 将新数据转换为 DataFrame
                new_data_df = pd.DataFrame([new_data])

                # 使用 pd.concat 而不是 append
                json_data_pd = pd.concat([json_data_pd, new_data_df], ignore_index=True)

        # Save the DataFrame to a variable for further use
        input_pd = json_data_pd

        if ref_paper_num>0:
                
            ## change col name
            input_pd['ref_title'] = [filename for filename in self.file_names]
            input_pd["ref_context"] = [""]*ref_paper_num
            input_pd["ref_entry"] = input_pd["reference paper citation information (can be collected from Google scholar/DBLP)"]
            input_pd["abstract"] = input_pd["reference paper abstract (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid abstract')
            input_pd["intro"] = input_pd["reference paper introduction (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid introduction')

            # optional columns
            input_pd["label"] = input_pd["reference paper category label (optional)"].apply(lambda x: str(x) if len(str(x))>0 else '')
            #input_pd["label"] = input_pd["reference paper category id (optional)"].apply(lambda x: str(x) if len(str(x))>0 else '')
            ## output tsv
            # output_tsv_filename = self.tsv_path + self.survey_id + '.tsv'
            output_tsv_filename = os.path.join(self.tsv_path, self.survey_id + '.tsv') 

            #output_df = input_pd[["ref_title","ref_context","ref_entry","abstract","intro","description"]]
            output_df = input_pd[["ref_title","ref_context","ref_entry","abstract","intro", 'label']]
            # print(output_df)

            #pdb.set_trace()
            output_df.to_csv(output_tsv_filename, sep='\t')

    def description_generation(self) -> None:
        query= self.cluster_standard
        query_list = generate_sentence_patterns(query)
        for name in self.collection_names:
            context, citation_data = query_embeddings_new_new(name, query_list)
            self.citation_data.extend(citation_data)

            description = generate(context, query, name)
            self.description_list.append(description)

        citation_path = f'{self.info_path}/{self.survey_id}/citation_data.json'
        os.makedirs(os.path.dirname(citation_path), exist_ok=True)
        with open(citation_path, 'w', encoding="utf-8") as outfile:
            json.dump(self.citation_data, outfile, indent=4, ensure_ascii=False)
        
        file_path = f'{self.tsv_path}/{self.survey_id}.tsv'

        with open(file_path, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')
            rows = list(reader)
        if rows:
            headers = rows[0]
            headers.append('retrieval_result')

            updated_rows = [headers]
            for row, description in zip(rows[1:], self.description_list):
                row.append(description)
                updated_rows.append(row)

            with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile, delimiter='\t')
                writer.writerows(updated_rows)

            print('Updated file has been saved to', file_path)
        else:
            print('Input file is empty.')

    def agglomerative_clustering(self) -> None:
        df = pd.read_csv(f'{self.tsv_path}/{self.survey_id}.tsv', sep='\t', index_col=0, encoding='utf-8')
        df_selected = df

        df_selected, _ = clustering(df_selected, 3, self.survey_id, self.info_path, self.tsv_path)
        self.df_selected = df_selected

        df_tmp = df_selected.reset_index()
        df_tmp['index'] = df_tmp.index
        ref_titles = list(df_tmp.groupby(df_tmp['label'])['ref_title'].apply(list))
        # ref_indexs = list(df_tmp.groupby(df_tmp['label'])['index'].apply(list))

        category_label_summarized = generate_cluster_name_new(f"{self.tsv_path}/{self.survey_id}.tsv", self.survey_title)
        self.cluster_names = category_label_summarized

        cluster_info = {category_label_summarized[i]:ref_titles[i] for i in range(len(category_label_summarized))}
        for key, value in cluster_info.items():
            temp = [legal_pdf(i) for i in value]
            cluster_info[key] = temp
            self.collection_names_clustered.append(temp)
        cluster_info_path = f'{self.info_path}/{self.survey_id}/cluster_info.json'
        with open(cluster_info_path, 'w', encoding="utf-8") as outfile:
            json.dump(cluster_info, outfile, indent=4, ensure_ascii=False)        

    def outline_generation(self) -> None:
        print(self.df_selected)
        print(self.cluster_names)
        outline_generator = OutlineGenerator(self.pipeline, self.df_selected, self.cluster_names)
        outline_generator.get_cluster_info()
        messages, outline = outline_generator.generate_outline_qwen(self.survey_title)
        outline_json = {'messages':messages, 'outline': outline}
        output_path = f'{self.info_path}/{self.survey_id}/outline.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding="utf-8") as outfile:
            json.dump(outline_json, outfile, indent=4, ensure_ascii=False)
        

    def section_generation(self) -> None:
        generateSurvey_qwen_new(self.survey_id, self.survey_title, self.collection_names_clustered, self.pipeline, self.citation_data, './txt','./info')

    def citation_generation(self) -> None:
        """
        Generate citation Markdown and PDF files from JSON and store them in the specified result path.
        """

        json_filepath = os.path.join(self.info_path, self.survey_id, "generated_result.json")

        markdown_dir = f'{self.result_path}/{self.survey_id}'
        markdown_filename = f'survey_{self.survey_id}.md'
        markdown_filepath = os.path.join(markdown_dir, markdown_filename)
        pdf_filename = f'survey_{self.survey_id}.pdf'
        pdf_filepath = os.path.join(markdown_dir, pdf_filename)

        markdown_content = self.get_markdown_content(json_filepath)
        if not markdown_content:
            raise ValueError("Markdown content is empty. Cannot generate citation files.")

        try:
            with open(markdown_filepath, 'w', encoding='utf-8', encoding="utf-8") as markdown_file:
                markdown_file.write(markdown_content)
            print(f"Markdown content saved to: {markdown_filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to save Markdown file: {e}")

        try:
            pdf = MarkdownPdf()
            pdf.meta["title"] = "Citation Results" 
            pdf.add_section(Section(markdown_content, toc=False))  
            pdf.save(pdf_filepath) 
            print(f"PDF content saved to: {pdf_filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate PDF file: {e}")
        print(f"Files generated successfully: \nMarkdown: {markdown_filepath}\nPDF: {pdf_filepath}")

    def get_markdown_content(self, json_filepath: str) -> str:
        """
        Read a JSON file and generate Markdown content based on its data.

        :param json_filepath: Path to the JSON file containing survey data.
        :return: A string containing the generated Markdown content.
        """
        try:
            with open(json_filepath, 'r', encoding='utf-8', encoding="utf-8") as json_file:
                survey_data = json.load(json_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read JSON file: {e}")

        topic = survey_data.get('survey_title', 'Default Topic')
        content = survey_data.get('content', 'No content available.')

        survey_title_markdown = f"# A Survey of {topic}\n\n"
        survey_content_markdown = content + "\n\n"

        markdown_content = survey_title_markdown + survey_content_markdown
        markdown_content = finalize_survey_paper(markdown_content, self.collection_names, self.file_names)
        return markdown_content

if __name__ == "__main__":
    root_path = "."
    pdf_path = "./pdfs/test"
    survey_title = "Automating Literature Review Generation with LLM"
    cluster_standard = "method"
    asg_system = ASG_system(root_path, 'test', pdf_path, survey_title, cluster_standard)
    asg_system.parsing_pdfs()
    asg_system.description_generation()
    asg_system.agglomerative_clustering()
    asg_system.outline_generation()
    asg_system.section_generation()
    asg_system.citation_generation()
