from __future__ import unicode_literals
import sys
from langchain_huggingface import HuggingFaceEmbeddings
# 禁用所有遥测功能
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['DISABLE_TELEMETRY'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
# 添加 Hugging Face 离线模式和缓存设置
# os.environ['HF_DATASETS_OFFLINE'] = '1'  
# os.environ['TRANSFORMERS_CACHE'] = './models/transformers_cache'
# os.environ['HF_HOME'] = './models/huggingface_cache'
# os.environ['HF_HUB_CACHE'] = './models/huggingface_hub_cache'

from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
import json
import requests
import time
import pandas as pd
import shutil
import traceback
from io import BytesIO

import hashlib
import re
import os
import csv
import xml.etree.ElementTree as ET
import urllib.parse

from django.http import JsonResponse
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage

# from .parse import DocumentLoading
from .asg_retriever import legal_pdf, process_pdf, query_embeddings_new_new, cleanup_retriever
from .asg_generator import generate,generate_sentence_patterns, cleanup_openai_client
from .asg_outline import OutlineGenerator,generateOutlineHTML_qwen, generateSurvey_qwen_new
from .asg_clustername import generate_cluster_name_new
from .postprocess import generate_references_section
from .asg_query import generate_generic_query_qwen, generate_query_qwen
from .asg_add_flowchart import insert_ref_images, detect_flowcharts
from .asg_mindmap import generate_graphviz_png, insert_outline_image
from .asg_latex import tex_to_pdf, insert_figures, md_to_tex, preprocess_md
# from .survey_generator_api import ensure_all_papers_cited
import glob

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps

load_dotenv()
# # 打印所有环境变量（可选，调试时使用）
# print("所有环境变量:", os.environ)

# # 获取特定环境变量
# openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_api_base = os.getenv("OPENAI_API_BASE")

# # 打印获取到的值
# print(f"OPENAI_API_KEY: {openai_api_key}")
# print(f"OPENAI_API_BASE: {openai_api_base}")

# 导入异步任务支持
from .middleware import async_task, task_manager

import os
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

DATA_PATH = './src/static/data/pdf/'
TXT_PATH = './src/static/data/txt/'
TSV_PATH = './src/static/data/tsv/'
MD_PATH = './src/static/data/md/'
INFO_PATH = './src/static/data/info/'
IMG_PATH = './src/static/img/'

paths = [DATA_PATH, TXT_PATH, TSV_PATH, MD_PATH, INFO_PATH, IMG_PATH]

for path in paths:
    path_obj = Path(path)
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")



Survey_dict = {
    '2742488' : 'Energy Efficiency in Cloud Computing',
    '2830555' : 'Cache Management for Real-Time Systems',
    '2907070' : 'Predictive Modeling on Imbalanced Data',
    '3073559' : 'Malware Detection with Data Mining',
    '3274658' : 'Analysis of Handwritten Signature'
}



Survey_Topic_dict = {
    '2742488' : ['energy'],
    '2830555' : ['cache'],
    '2907070' : ['imbalanced'],
    '3073559' : ['malware', 'detection'],
    '3274658' : ['handwritten', 'signature']
}


Survey_n_clusters = {
    '2742488' : 3,
    '2830555' : 3,
    '2907070' : 3,
    '3073559' : 3,
    '3274658' : 2
}

Global_survey_id = ""
Global_survey_title=""
Global_ref_list = []
Global_category_description = []
Global_category_label = []
Global_df_selected = ""
Global_test_flag = False
Global_collection_names = []
Global_collection_names_clustered = []
Global_file_names=[]
Global_description_list = []
Global_cluster_names = []
Global_citation_data = []
Global_cluster_num = 4


# 创建模型缓存目录
import os
from pathlib import Path

def ensure_cache_dirs():
    """确保缓存目录存在"""
    cache_dirs = [
        './models/transformers_cache',
        './models/huggingface_cache', 
        './models/huggingface_hub_cache'
    ]
    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

# def init_embedder_with_retry():
#     """初始化embedder，带重试和错误处理"""
#     ensure_cache_dirs()
    
#     try:
#         print("正在初始化 SentenceTransformer embeddings...")
#         # 尝试初始化embedder
#         model = SentenceTransformer(
#             'sentence-transformers/all-MiniLM-L6-v2',
#             cache_folder='./models/transformers_cache'
#         )
#         print("SentenceTransformer embeddings 初始化成功")
#         return model
        
#     except Exception as e:
#         print(f"初始化 SentenceTransformer embeddings 失败: {e}")
#         print("尝试使用本地缓存或替代方案...")
        
#         try:
#             # 尝试使用本地缓存
#             model = SentenceTransformer(
#                 'sentence-transformers/all-MiniLM-L6-v2',
#                 cache_folder='./models/transformers_cache',
#                 local_files_only=True
#             )
#             print("使用本地缓存成功")
#             return model
#         except Exception as e2:
#             print(f"使用本地缓存也失败: {e2}")
#             print("警告: 将使用空的 embedder，某些功能可能不可用")
#             return None

# 初始化embedder
embedder = None  # 延迟初始化

def get_embedder():
    """获取embedder实例，如果未初始化则进行初始化"""
    global embedder
    if embedder is None:
        try:
            print("正在初始化 embedder...")
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("embedder 初始化完成")
        except Exception as e:
            print(f"embedder 初始化失败: {e}")
            return None
    return embedder

from demo.category_and_tsne import clustering

# 添加超时装饰器
def timeout_handler(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                # 线程仍在运行，说明超时了
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator

# 添加进度跟踪
progress_tracker = {}

def update_progress(operation_id, progress, message=""):
    """更新操作进度"""
    progress_tracker[operation_id] = {
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }
    print(f"[{operation_id}] {progress}% - {message}")

def get_progress(operation_id):
    """获取操作进度"""
    return progress_tracker.get(operation_id, {'progress': 0, 'message': 'Starting...', 'timestamp': time.time()})

# 添加进度查询端点
@csrf_exempt
def get_operation_progress(request):
    """获取操作进度的API端点，支持异步任务结果"""
    if request.method == 'GET':
        operation_id = request.GET.get('operation_id')
        if operation_id:
            print(f"[DEBUG] Checking progress for operation_id: {operation_id}")
            
            # 首先检查异步任务状态
            task_status = task_manager.get_task_status(operation_id)
            print(f"[DEBUG] Task status: {task_status}")
            
            if task_status['status'] == 'completed':
                # 任务完成，返回结果
                print(f"[DEBUG] Task {operation_id} completed, returning result")
                result = task_status.get('result')
                
                # 检查result是否是HttpResponse对象（旧的PDF生成方式）
                if hasattr(result, 'content'):
                    try:
                        import json
                        content = json.loads(result.content.decode('utf-8'))
                        return JsonResponse({
                            'progress': 100,
                            'message': 'Task completed successfully!',
                            'status': 'completed',
                            'result': content
                        })
                    except Exception as e:
                        print(f"[DEBUG] Error parsing HttpResponse content: {e}")
                        # 对于PDF等二进制文件，我们不解析内容，只返回完成状态
                        return JsonResponse({
                            'progress': 100,
                            'message': 'Task completed successfully!',
                            'status': 'completed',
                            'result': {'message': 'Binary file generated successfully'}
                        })
                # 检查result是否是Django JsonResponse对象
                elif hasattr(result, 'content') and hasattr(result, 'status_code'):
                    try:
                        import json
                        content = json.loads(result.content.decode('utf-8'))
                        return JsonResponse({
                            'progress': 100,
                            'message': 'Task completed successfully!',
                            'status': 'completed',
                            'result': content
                        })
                    except Exception as e:
                        print(f"[DEBUG] Error parsing JsonResponse content: {e}")
                        return JsonResponse({
                            'progress': 100,
                            'message': 'Task completed successfully!',
                            'status': 'completed'
                        })
                else:
                    # 普通的结果对象
                    return JsonResponse({
                        'progress': 100,
                        'message': 'Task completed successfully!',
                        'status': 'completed',
                        'result': result
                    })
            elif task_status['status'] == 'failed':
                # 任务失败
                print(f"[DEBUG] Task {operation_id} failed: {task_status.get('error')}")
                return JsonResponse({
                    'progress': -1,
                    'message': f"Task failed: {task_status.get('error', 'Unknown error')}",
                    'status': 'failed',
                    'error': task_status.get('error')
                })
            elif task_status['status'] == 'running':
                # 任务正在运行，返回当前进度
                print(f"[DEBUG] Task {operation_id} is running, checking progress")
                progress_info = get_progress(operation_id)
                print(f"[DEBUG] Progress info: {progress_info}")
                return JsonResponse({
                    **progress_info,
                    'status': 'running'
                })
            else:
                # 任务未找到，返回默认进度
                print(f"[DEBUG] Task {operation_id} not found, returning default progress")
                progress_info = get_progress(operation_id)
                return JsonResponse({
                    **progress_info,
                    'status': 'not_found'
                })
        return JsonResponse({'error': 'operation_id is required'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

class reference_collection(object):
    def __init__(
            self,
            input_df
    ):
        self.input_df = input_df

    def full_match_with_entries_in_pd(self, query_paper_titles):
        entries_in_pd = self.input_df.copy()
        entries_in_pd['ref_title'] = entries_in_pd['ref_title'].apply(str.lower)
        query_paper_titles = [i.lower() for i in query_paper_titles]

        # matched_entries = entries_in_pd[entries_in_pd['ref_title'].isin(query_paper_titles)]
        matched_entries = self.input_df[entries_in_pd['ref_title'].isin(query_paper_titles)]
        return matched_entries,matched_entries.shape[0]

    # select the sentences that can match with the topic words
    def match_ref_paper(self, query_paper_titles,match_mode='full', match_ratio=70):
        # query_paper_title = query_paper_title.lower()
        # two modes for str matching
        if match_mode == 'full':
            matched_entries, matched_num = self.full_match_with_entries_in_pd(query_paper_titles)
        return matched_entries, matched_num


def generate_uid():
    uid_str=""
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode('utf-8'))
    uid_str= hash.hexdigest()[:10]

    return uid_str

def index(request):
    return render(request, 'demo/index.html')

def delete_files(request):
    if request.method == 'POST':
        try:
            folders = ['./src/static/data/pdf/', './src/static/data/tsv/', './src/static/data/txt/', './src/static/data/md/']
            for folder in folders:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        return JsonResponse({'success': False, 'message': str(e)})
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})

    return JsonResponse({'success': False, 'message': 'Invalid request method'})

def clean_str(input_str):
    input_str = str(input_str).strip().lower()
    if input_str == "none" or input_str == "nan" or len(input_str) == 0:
        return ""
    input_str = input_str.replace('\\n',' ').replace('\n',' ').replace('\r',' ').replace('——',' ').replace('——',' ').replace('__',' ').replace('__',' ').replace('........','.').replace('....','.').replace('....','.').replace('..','.').replace('..','.').replace('..','.').replace('. . . . . . . . ','. ').replace('. . . . ','. ').replace('. . . . ','. ').replace('. . ','. ').replace('. . ','. ')
    input_str = re.sub(r'\\u[0-9a-z]{4}', ' ', input_str).replace('  ',' ').replace('  ',' ')
    return input_str

def PosRank_get_top5_ngrams(input_pd):
    pos = {'NOUN', 'PROPN', 'ADJ'}
    extractor = PosRank()

    abs_top5_unigram_list_list = []
    abs_top5_bigram_list_list = []
    abs_top5_trigram_list_list = []
    intro_top5_unigram_list_list = []
    intro_top5_bigram_list_list = []
    intro_top5_trigram_list_list = []

    for line_index,pd_row in input_pd.iterrows():

        input_str=pd_row["abstract"].replace('-','')
        extractor.load_document(input=input_str,language='en',normalization=None)

        #unigram
        unigram_extractor=extractor
        unigram_extractor.candidate_selection(maximum_word_number=1,minimum_word_number=1)
        unigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_unigram_list = []
        for (keyphrase, score) in unigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_unigram_list.append(keyphrase)
        #pdb.set_trace()
        #bigram
        bigram_extractor=extractor
        bigram_extractor.candidate_selection(maximum_word_number=2,minimum_word_number=2)
        bigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_bigram_list = []
        for (keyphrase, score) in bigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_bigram_list.append(keyphrase)

        #trigram
        trigram_extractor=extractor
        trigram_extractor.candidate_selection(maximum_word_number=3,minimum_word_number=3)
        trigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_trigram_list = []
        for (keyphrase, score) in trigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_trigram_list.append(keyphrase)

        abs_top5_unigram_list_list.append(abs_top5_unigram_list)
        abs_top5_bigram_list_list.append(abs_top5_bigram_list)
        abs_top5_trigram_list_list.append(abs_top5_trigram_list)

    return abs_top5_unigram_list_list,abs_top5_bigram_list_list,abs_top5_trigram_list_list

def process_file(file_name, survey_id, mode):
    embedder_instance = get_embedder()
    if embedder_instance is None:
        print("警告: embedder 未初始化，跳过PDF处理")
        # 返回一个默认值或抛出更友好的错误
        collection_name = f"collection_{survey_id}_{int(time.time())}"
        name = file_name.split('/')[-1].replace('.pdf', '')
        return collection_name, name
    
    result = process_pdf(file_name, survey_id, embedder_instance, mode)
    collection_name = result[0]
    name = result[-1]
    return collection_name, name

def sanitize_filename_py(filename):
    last_dot = filename.rfind('.')
    
    def sanitize_part(part):
        part = part.lower()
        part = re.sub(r'[^a-z0-9]', ' ', part)
        part = re.sub(r'\s+', ' ', part)
        part = part.strip()
        words = part.split(' ')        
        if len(words) == 0:
            return ''  
        words[0] = words[0].capitalize()
        
        return ' '.join(words)
    
    if last_dot == -1:
        # No extension
        return sanitize_part(filename)
    elif last_dot == 0:
        # Hidden file
        extension = filename[1:]
        return '.' + sanitize_part(extension)
    else:
        # With extension
        name = filename[:last_dot]
        extension = filename[last_dot + 1:]
        return sanitize_part(name) + '.' + sanitize_part(extension)

def get_existing_survey_ids():

    tsv_directory = os.path.join("src", "static", "data", "tsv")
    survey_ids = []
    try:
        for file_name in os.listdir(tsv_directory):
            if file_name.endswith(".tsv"):
                # 去掉 .tsv 后缀
                survey_ids.append(file_name[:-4])
    except Exception as e:
        print("Error reading tsv directory:", e)
    return survey_ids

def get_surveys(request):

    surveys = get_existing_survey_ids()
    return JsonResponse({'surveys': surveys})

@csrf_exempt
@timeout_handler(1800)  # 15分钟超时
def upload_refs_sync(request):
    """同步版本的文件上传处理函数"""
    start_time = time.time()
    operation_id = getattr(request, 'operation_id', f"upload_{int(start_time)}")
    print(f"[DEBUG] upload_refs_sync started with operation_id: {operation_id}")
    update_progress(operation_id, 0, "Starting file upload...")
    
    RECOMMENDED_PDF_DIR = os.path.join("src", "static", "data", "pdf", "recommend_pdfs")
    if request.method == 'POST':
        try:
            # 检查是否有上传的文件（新方式使用file_paths，旧方式使用FILES）
            has_uploaded_files = False
            if hasattr(request, 'file_paths') and request.file_paths:
                has_uploaded_files = True
            elif request.FILES:
                has_uploaded_files = True
            
            if not has_uploaded_files:
                if not os.path.exists(RECOMMENDED_PDF_DIR):
                    return JsonResponse({'error': 'No file part'}, status=400)
        
            update_progress(operation_id, 10, "Initializing upload process...")
            
            is_valid_submission = True
            has_label_id = False
            has_ref_link = False

            filenames = []
            collection_names = []
            filesizes = []
            
            # 创建统一的file_dict，兼容新旧两种方式
            file_dict = {}
            
            # 如果使用新的file_paths方式
            if hasattr(request, 'file_paths') and request.file_paths:
                update_progress(operation_id, 15, "Loading files from disk...")
                for file_path in request.file_paths:
                    file_name = os.path.basename(file_path)
                    
                    # 从磁盘读取文件内容
                    with open(file_path, 'rb') as f:
                        file_content = BytesIO(f.read())
                    
                    # 创建类似Django上传文件的对象
                    uploaded_file = InMemoryUploadedFile(
                        file_content,
                        field_name="file",
                        name=file_name,
                        content_type="application/pdf",
                        size=os.path.getsize(file_path),
                        charset=None
                    )
                    file_dict[file_name] = uploaded_file
            else:
                # 使用传统的request.FILES方式
                file_dict = request.FILES.copy()

            global Global_survey_id
            global Global_test_flag
            global Global_collection_names
            global Global_survey_title
            global Global_file_names

            Global_survey_title = request.POST.get('topic', False)
            process_pdf_mode = request.POST.get('mode', False)
            
            update_progress(operation_id, 20, "Processing recommended PDFs...")
            
            if os.path.exists(RECOMMENDED_PDF_DIR):
                for pdf_name in os.listdir(RECOMMENDED_PDF_DIR):
                    if pdf_name.endswith(".pdf"):
                        pdf_path = os.path.join(RECOMMENDED_PDF_DIR, pdf_name)

                        pdf_content = BytesIO()
                        with open(pdf_path, 'rb') as f:
                            shutil.copyfileobj(f, pdf_content)
                        pdf_content.seek(0)

                        uploaded_pdf = InMemoryUploadedFile(
                            pdf_content,
                            field_name="file",
                            name=pdf_name,
                            content_type="application/pdf",
                            size=os.path.getsize(pdf_path),
                            charset=None
                        )

                        file_dict[f"recommend_{pdf_name}"] = uploaded_pdf

                shutil.rmtree(RECOMMENDED_PDF_DIR)

            update_progress(operation_id, 30, "Setting up survey ID...")
            
            # 始终生成新的survey_id，无论前端传递什么参数
            Global_survey_id = 'test_4' if Global_test_flag else generate_uid()
            uid_str = Global_survey_id
            print(f"[DEBUG] Generated new survey_id: {Global_survey_id}")

            update_progress(operation_id, 40, "Processing uploaded files...")
            
            total_files = len(file_dict)
            processed_files = 0
            
            for file_name in file_dict:
                file = file_dict[file_name]
                if not file.name:
                    continue
                if file:
                    try:
                        sanitized_filename = sanitize_filename_py(os.path.splitext(file.name)[0])
                        file_extension = os.path.splitext(file.name)[1].lower()
                        if sanitized_filename in filenames:
                            continue
                        sanitized_filename = f"{sanitized_filename}{file_extension}"

                        file_path = os.path.join('src', 'static', 'data', 'pdf', Global_survey_id, sanitized_filename)
                        if default_storage.exists(file_path):
                            default_storage.delete(file_path)
                        
                        saved_file_name = default_storage.save(file_path, file)
                        file_size = round(float(file.size) / 1024000, 2)

                        collection_name, processed_file = process_file(saved_file_name, Global_survey_id, process_pdf_mode)
                        Global_collection_names.append(collection_name)
                        Global_file_names.append(processed_file)
                        filenames.append(processed_file)
                        filesizes.append(file_size)
                        
                        processed_files += 1
                        progress = 40 + (processed_files / total_files) * 30
                        update_progress(operation_id, progress, f"Processed {processed_files}/{total_files} files")
                        
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")
                        continue

            update_progress(operation_id, 70, "Generating JSON data...")
            
            new_file_name = Global_survey_id
            csvfile_name = new_file_name + '.'+ file_name.split('.')[-1]

            json_data_pd = pd.DataFrame()
            json_files_path = f'./src/static/data/txt/{Global_survey_id}/*.json'
            json_files = glob.glob(json_files_path)

            # Dictionary to hold title and abstract pairs
            title_abstract_dict = {}
            filtered_json_files = [
                json_file for json_file in json_files
                if os.path.splitext(os.path.basename(json_file))[0] in filenames
            ]
            ref_paper_num = len(filtered_json_files)
            print(f'The length of the json files is {ref_paper_num}')

            update_progress(operation_id, 80, "Processing JSON files...")
            
            # Iterate over each JSON file
            for i, file_path in enumerate(filtered_json_files):
                try:
                    with open(file_path, 'r', encoding= "utf-8") as file:
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

                        new_data_df = pd.DataFrame([new_data])
                        json_data_pd = pd.concat([json_data_pd, new_data_df], ignore_index=True)
                        title_abstract_dict[title] = abstract
                        
                        progress = 80 + (i / len(filtered_json_files)) * 10
                        update_progress(operation_id, progress, f"Processing JSON {i+1}/{len(filtered_json_files)}")
                        
                except Exception as e:
                    print(f"Error processing JSON file {file_path}: {e}")
                    continue

            update_progress(operation_id, 90, "Finalizing data...")
            
            input_pd = json_data_pd
            output_path = f'./src/static/data/info/{Global_survey_id}/title_abstract_pairs.json'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding="utf-8") as outfile:
                json.dump(title_abstract_dict, outfile, indent=4, ensure_ascii=False)

            print(f'Title-abstract pairs have been saved to {output_path}')

            # 初始化 output_tsv_filename，确保它总是被定义
            output_tsv_filename = "./src/static/data/tsv/" + new_file_name + '.tsv'

            if ref_paper_num>0:

                print('The filenames are:', filenames)
                print('The json files are:', filtered_json_files)
                input_pd['ref_title'] = [filename for filename in filenames]
                input_pd["ref_context"] = [""]*ref_paper_num
                input_pd["ref_entry"] = input_pd["reference paper citation information (can be collected from Google scholar/DBLP)"]
                input_pd["abstract"] = input_pd["reference paper abstract (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid abstract')
                input_pd["intro"] = input_pd["reference paper introduction (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid introduction')

                input_pd["label"] = input_pd["reference paper category label (optional)"].apply(lambda x: str(x) if len(str(x))>0 else '')

                try:
                    output_df = input_pd[["ref_title","ref_context","ref_entry","abstract","intro"]]

                    if has_label_id == True:
                        output_df["label"]=input_pd["label"]
                    else:
                        output_df["label"]=[""]*input_pd.shape[0]

                    output_df.to_csv(output_tsv_filename, sep='\t')
                except Exception as e:
                    print(f"Cannot output tsv: {e}")
                    is_valid_submission = False

            else:
                is_valid_submission = False

            update_progress(operation_id, 100, "Upload completed successfully!")
            
            if is_valid_submission == True:
                ref_ids = [i for i in range(output_df['ref_title'].shape[0])]
                ref_list = {
                            'ref_ids':ref_ids,
                            'is_valid_submission':is_valid_submission,
                            "uid":uid_str,
                            "tsv_filename":output_tsv_filename,
                            # 'topic_words': clusters_topic_words,
                            'filenames': filenames,
                            'filesizes': filesizes,
                            'survey_id': Global_survey_id,
                            'operation_id': operation_id
                            }

            else:
                ref_list = {'ref_ids':[],'is_valid_submission':is_valid_submission,"uid":uid_str,"tsv_filename":output_tsv_filename, 'filenames': filenames, 'filesizes': filesizes, 'survey_id': Global_survey_id, 'operation_id': operation_id}
            ref_list = json.dumps(ref_list)
            print("--- %s seconds used in processing files ---" % (time.time() - start_time))
            
            # 清理临时文件（如果使用了file_paths方式）
            if hasattr(request, 'file_paths') and request.file_paths:
                try:
                    temp_dir = os.path.dirname(request.file_paths[0])
                    if 'tmp_upload' in temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f"[DEBUG] Cleaned up temporary upload directory: {temp_dir}")
                except Exception as e:
                    print(f"[DEBUG] Failed to clean up temporary directory: {e}")
            
            return HttpResponse(ref_list)
            
        except TimeoutError as e:
            update_progress(operation_id, -1, f"Upload timed out: {str(e)}")
            # 清理临时文件
            if hasattr(request, 'file_paths') and request.file_paths:
                try:
                    temp_dir = os.path.dirname(request.file_paths[0])
                    if 'tmp_upload' in temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except:
                    pass
            return JsonResponse({'error': f'Upload operation timed out: {str(e)}'}, status=408)
        except Exception as e:
            update_progress(operation_id, -1, f"Upload failed: {str(e)}")
            # 清理临时文件
            if hasattr(request, 'file_paths') and request.file_paths:
                try:
                    temp_dir = os.path.dirname(request.file_paths[0])
                    if 'tmp_upload' in temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except:
                    pass
            return JsonResponse({'error': f'Upload failed: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def generate_arxiv_query(request):
    def search_arxiv_with_query(query, max_results=50):
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate"
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching data with query: {query} | status code: {response.status_code}")
            return []
        
        try:
            root = ET.fromstring(response.text)
        except Exception as e:
            print("Error parsing XML:", e)
            return []
        
        ns = "{http://www.w3.org/2005/Atom}"
        entries = root.findall(f"{ns}entry")
        papers = []
        for entry in entries:
            title_elem = entry.find(f"{ns}title")
            title = title_elem.text.strip() if title_elem is not None else ""
            summary_elem = entry.find(f"{ns}summary")
            summary_text = summary_elem.text.strip() if summary_elem is not None else ""
            link_elem = entry.find(f"{ns}id")
            link_text = link_elem.text.strip() if link_elem is not None else ""
            arxiv_id = link_text.split('/')[-1]
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            papers.append({
                "title": title,
                "summary": summary_text,
                "pdf_link": pdf_link,
                "arxiv_id": arxiv_id
            })
        
        return papers
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            topic = data.get('topic', '').strip()
            if not topic:
                return JsonResponse({'error': 'Topic is required.'}, status=400)
            max_results = 50
            min_results = 10
            strict_query = generate_query_qwen(topic)
            papers_strict = search_arxiv_with_query(strict_query, max_results=max_results)

            total_papers = {paper["title"]: paper for paper in papers_strict}

            if len(total_papers) >= min_results:
                papers_list = list(total_papers.values())  # dict -> list

                return JsonResponse({
                    "papers": papers_list,  # 例如 [{"title": "...", "summary": "...", "pdf_link": "...", "arxiv_id": "..."}]
                    "count": len(papers_list),
                }, status=200)

            attempts = 0
            MAX_ATTEMPTS = 5
            current_query = strict_query  # 方便追踪当前 query

            while len(total_papers) < min_results and attempts < MAX_ATTEMPTS:
                # 生成更宽松的查询
                generic_query = generate_generic_query_qwen(current_query, topic)
                papers_generic = search_arxiv_with_query(generic_query, max_results=max_results)

                # 合并新结果
                new_count = 0
                for paper in papers_generic:
                    if paper["title"] not in total_papers:
                        total_papers[paper["title"]] = paper
                        new_count += 1

                attempts += 1
                current_query = generic_query  # 将本轮的宽松查询作为"新的严格查询"

                if len(total_papers) >= min_results:
                    # 一旦达到 min_results，就返回此时的查询
                    papers_list = list(total_papers.values())  # dict -> list

                    return JsonResponse({
                        "papers": papers_list,  # 例如 [{"title": "...", "summary": "...", "pdf_link": "...", "arxiv_id": "..."}]
                        "count": len(papers_list),
                    }, status=200)

            return JsonResponse({
                'error': f'Not enough references found even after {attempts} attempts.',
                'count': len(total_papers),
            }, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)

@csrf_exempt
@timeout_handler(900)  # 15分钟超时
def download_pdfs_sync(request):
    """同步版本的PDF下载函数"""
    def clean_filename(filename):
        filename = filename.strip()  # 去掉首尾空格和换行符
        filename = re.sub(r'[\\/*?:"<>|\n\r]', '', filename)  # 移除非法字符
        return filename
    
    start_time = time.time()
    operation_id = f"download_{int(start_time)}"
    print(f"[DEBUG] download_pdfs_sync started with operation_id: {operation_id}")
    update_progress(operation_id, 0, "Starting PDF downloads...")
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            pdf_links = data.get("pdf_links", [])
            pdf_titles = data.get("pdf_titles", [])  # PDF 标题列表
            print(f"Starting download of {len(pdf_links)} PDFs")

            if not pdf_links:
                update_progress(operation_id, -1, "No PDFs to download")
                return JsonResponse({"message": "No PDFs to download."}, status=400)

            base_dir = os.path.join(os.getcwd(), "src", "static", "data", "pdf", "recommend_pdfs")
            os.makedirs(base_dir, exist_ok=True)  # 确保文件夹存在

            downloaded_files = []
            failed_downloads = []
            
            update_progress(operation_id, 10, f"Preparing to download {len(pdf_links)} PDFs...")
            
            for i, pdf_url in enumerate(pdf_links):
                try:
                    print(f"Downloading {i+1}/{len(pdf_links)}: {pdf_url}")
                    progress = 10 + (i / len(pdf_links)) * 80
                    update_progress(operation_id, progress, f"Downloading PDF {i+1}/{len(pdf_links)}")
                    
                    # 设置超时时间：连接超时10秒，读取超时60秒
                    response = requests.get(
                        pdf_url, 
                        stream=True, 
                        timeout=(10, 60),
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    )
                    
                    if response.status_code == 200:
                        # 处理文件名，确保合法
                        sanitized_title = clean_filename(pdf_titles[i]) if i < len(pdf_titles) else f"file_{i}"
                        pdf_filename = os.path.join(base_dir, f"{sanitized_title}.pdf")

                        # 下载 PDF，添加文件大小检查
                        total_size = 0
                        max_size = 50 * 1024 * 1024  # 50MB 限制
                        
                        with open(pdf_filename, "wb") as pdf_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    total_size += len(chunk)
                                    if total_size > max_size:
                                        print(f"File too large, skipping: {pdf_url}")
                                        failed_downloads.append({"url": pdf_url, "reason": "File too large (>50MB)"})
                                        break
                                    pdf_file.write(chunk)
                        
                        if total_size <= max_size:
                            downloaded_files.append(pdf_filename)
                            print(f"Success: {pdf_filename} ({total_size/1024/1024:.2f}MB)")
                        else:
                            # 删除部分下载的文件
                            if os.path.exists(pdf_filename):
                                os.remove(pdf_filename)
                    else:
                        print(f"Failed to download {pdf_url}, status code: {response.status_code}")
                        failed_downloads.append({"url": pdf_url, "reason": f"HTTP {response.status_code}"})

                except requests.exceptions.Timeout:
                    print(f"Timeout downloading {pdf_url}")
                    failed_downloads.append({"url": pdf_url, "reason": "Timeout"})
                except requests.exceptions.ConnectionError:
                    print(f"Connection error downloading {pdf_url}")
                    failed_downloads.append({"url": pdf_url, "reason": "Connection error"})
                except Exception as e:
                    print(f"Error downloading {pdf_url}: {e}")
                    failed_downloads.append({"url": pdf_url, "reason": str(e)})

            print(f"Download finished: {len(downloaded_files)} successful, {len(failed_downloads)} failed")
            update_progress(operation_id, 100, f"Download completed: {len(downloaded_files)} successful, {len(failed_downloads)} failed")
            
            # 构建响应消息
            message = f"Downloaded {len(downloaded_files)} PDFs successfully!"
            if failed_downloads:
                message += f" {len(failed_downloads)} downloads failed."
            
            result = {
                "message": message,
                "files": downloaded_files,
                "failed": failed_downloads,
                "success_count": len(downloaded_files),
                "total_count": len(pdf_links)
            }
            
            return JsonResponse(result)

        except json.JSONDecodeError:
            update_progress(operation_id, -1, "Invalid JSON data")
            return JsonResponse({"message": "Invalid JSON data."}, status=400)
        except Exception as e:
            print(f"Unexpected error in download_pdfs: {e}")
            update_progress(operation_id, -1, f"Error: {str(e)}")
            return JsonResponse({"message": "An error occurred.", "error": str(e)}, status=500)

    return JsonResponse({"message": "Invalid request method."}, status=405)

@csrf_exempt
def download_pdfs(request):
    """异步版本的PDF下载接口，立即返回operation_id避免Cloudflare 524超时"""
    if request.method == "POST":
        # 生成操作ID
        operation_id = f"download_{int(time.time())}"
        
        print(f"[DEBUG] Starting async download task: {operation_id}")
        
        # 启动异步任务
        success = task_manager.start_task(
            operation_id, 
            download_pdfs_sync, 
            request
        )
        
        if not success:
            print(f"[DEBUG] Task {operation_id} already running")
            return JsonResponse({'error': 'Download task already running'}, status=409)
        
        print(f"[DEBUG] Async task {operation_id} started successfully")
        
        # 立即返回operation_id，不等待处理完成
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'PDF download started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def annotate_categories(request):
    html = generateOutlineHTML_qwen(Global_survey_id)
    print("The outline has been parsed successfully.")
    return JsonResponse({'html': html})

@csrf_exempt
def get_topic(request):
    topic = request.POST.get('topics', False)
    references, ref_links, ref_ids = get_refs(topic)
    global Global_survey_id
    Global_survey_id = topic
    ref_list = {
        'references' : references,
        'ref_links'  : ref_links,
        'ref_ids'    : ref_ids
    }
    ref_list = json.dumps(ref_list)
    return HttpResponse(ref_list)

@csrf_exempt
@timeout_handler(1800)  # 30分钟超时
def automatic_taxonomy_sync(request):
    """同步版本的自动分类函数"""
    start_time = time.time()
    operation_id = f"taxonomy_{int(start_time)}"
    update_progress(operation_id, 0, "Starting automatic taxonomy...")
    
    global Global_description_list, Global_df_selected, Global_cluster_names, Global_ref_list, Global_category_label, Global_collection_names_clustered, Global_cluster_num
    global Global_survey_id, Global_collection_names, Global_citation_data, Global_file_names, Global_survey_title
    
    try:
        update_progress(operation_id, 10, "Loading reference data...")
        
        if request.method == 'POST':
            # 处理不同的请求格式
            try:
                # 尝试解析 JSON 数据
                if request.content_type == 'application/json':
                    data = json.loads(request.body)
                    Global_cluster_num = data.get('Global_cluster_num', 5)
                    refs_data = data.get('refs', [])
                    query = data.get('taxonomy_standard', '')
                else:
                    # 处理 form-data 格式
                    Global_cluster_num = int(request.POST.get('Global_cluster_num', 5))
                    refs_data = request.POST.get('refs', '[]')
                    query = request.POST.get('taxonomy_standard', '')
                    
                    # 解析 refs 数据
                    if isinstance(refs_data, str):
                        try:
                            refs_data = json.loads(refs_data)
                        except json.JSONDecodeError:
                            refs_data = []
                    
            except (json.JSONDecodeError, ValueError) as e:
                # 如果解析失败，尝试 form-data
                try:
                    Global_cluster_num = int(request.POST.get('Global_cluster_num', 5))
                    refs_data = request.POST.get('refs', '[]')
                    query = request.POST.get('taxonomy_standard', '')
                    
                    # 解析 refs 数据
                    if isinstance(refs_data, str):
                        try:
                            refs_data = json.loads(refs_data)
                        except json.JSONDecodeError:
                            refs_data = []
                except Exception as parse_error:
                    print(f"Error parsing request data: {parse_error}")
                    return JsonResponse({'error': f'Invalid request format: {parse_error}'}, status=400)
            
            # 处理 refs 数据
            if isinstance(refs_data, list):
                ref_list = [int(item) for item in refs_data if str(item).isdigit()]
            else:
                ref_list = []
            
            print(f"Parsed ref_list: {ref_list}")
            print(f"Global_cluster_num: {Global_cluster_num}")
            print(f"Query: {query}")
            
            update_progress(operation_id, 20, "Generating query patterns...")
            
            # 生成查询模式
            query_list = generate_sentence_patterns(query)
            
            update_progress(operation_id, 30, "Processing collections...")
            
            # 处理每个集合
            for name in Global_collection_names:
                context, citation_data = query_embeddings_new_new(name, query_list)
                Global_citation_data.extend(citation_data)
                
                description = generate(context, query, name)
                Global_description_list.append(description)
            
            update_progress(operation_id, 50, "Saving citation data...")
            
            # 保存引用数据
            citation_path = f'./src/static/data/info/{Global_survey_id}/citation_data.json'
            os.makedirs(os.path.dirname(citation_path), exist_ok=True)
            with open(citation_path, 'w', encoding="utf-8") as outfile:
                json.dump(Global_citation_data, outfile, indent=4, ensure_ascii=False)
            
            update_progress(operation_id, 60, "Updating TSV file...")
            
            # 更新 TSV 文件
            file_path = f'./src/static/data/tsv/{Global_survey_id}.tsv'
            with open(file_path, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile, delimiter='\t')
                rows = list(reader)
            
            if rows:
                headers = rows[0]
                headers.append('retrieval_result')
                
                updated_rows = [headers]
                for row, description in zip(rows[1:], Global_description_list):
                    row.append(description)
                    updated_rows.append(row)
                
                with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile, delimiter='\t')
                    writer.writerows(updated_rows)
                
                print('Updated file has been saved to', file_path)
            else:
                print('Input file is empty.')
            
            Global_ref_list = ref_list
            
            print('Categorization survey id', Global_survey_id)
            
            update_progress(operation_id, 70, "Performing clustering...")
            
            # 执行聚类
            colors, category_label = Clustering_refs(n_clusters=Global_cluster_num)
            Global_category_label = category_label
            
            update_progress(operation_id, 80, "Processing cluster results...")
            
            # 处理聚类结果
            df_tmp = Global_df_selected.reset_index()
            df_tmp['index'] = df_tmp.index
            ref_titles = list(df_tmp.groupby(df_tmp['label'])['ref_title'].apply(list))
            ref_indexs = list(df_tmp.groupby(df_tmp['label'])['index'].apply(list))
            
            # 读取主题信息
            info = pd.read_json(f'./src/static/data/info/{Global_survey_id}/topic.json')
            category_label = info['KeyBERT'].to_list()
            category_label_summarized = []
            
            tsv_path = f'./src/static/data/tsv/{Global_survey_id}.tsv'
            
            cluster_num = Global_cluster_num
            category_label_summarized = generate_cluster_name_new(tsv_path, Global_survey_title, cluster_num)
            Global_cluster_names = category_label_summarized
            
            update_progress(operation_id, 90, "Generating final results...")
            
            # 准备返回数据
            cate_list = {
                'colors': colors,
                'category_label': category_label_summarized,
                'survey_id': Global_survey_id,
                'ref_titles': [[i.title() for i in j] for j in ref_titles],
                'ref_indexs': ref_indexs
            }
            print(cate_list)
            
            # 保存聚类信息
            cluster_info = {category_label_summarized[i]: ref_titles[i] for i in range(len(category_label_summarized))}
            for key, value in cluster_info.items():
                temp = [legal_pdf(i) for i in value]
                cluster_info[key] = temp
                Global_collection_names_clustered.append(temp)
            
            cluster_info_path = f'./src/static/data/info/{Global_survey_id}/cluster_info.json'
            with open(cluster_info_path, 'w', encoding="utf-8") as outfile:
                json.dump(cluster_info, outfile, indent=4, ensure_ascii=False)
            
            # 生成大纲
            outline_generator = OutlineGenerator(Global_df_selected, Global_cluster_names)
            outline_generator.get_cluster_info()
            messages, outline = outline_generator.generate_outline_qwen(Global_survey_title, Global_cluster_num)
            
            outline_json = {'messages': messages, 'outline': outline}
            output_path = TXT_PATH + Global_survey_id + '/outline.json'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding="utf-8") as outfile:
                json.dump(outline_json, outfile, indent=4, ensure_ascii=False)
            
            update_progress(operation_id, 100, "Automatic taxonomy completed successfully!")
            
            # 返回 JSON 字符串格式，与原函数保持一致
            cate_list_json = json.dumps(cate_list)
            return HttpResponse(cate_list_json)
        
        else:
            return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    except Exception as e:
        print(f"Error in automatic_taxonomy: {str(e)}")
        import traceback
        traceback.print_exc()
        update_progress(operation_id, -1, f"Error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def save_updated_cluster_info(request):
    global Global_collection_names
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            survey_id = Global_survey_id
            updated_cate_list = data.get('updated_cate_list')
            ref_indexs = updated_cate_list.get("ref_indexs", [])
            rearranged_collection_names = [
                [Global_collection_names[index] for index in group] for group in ref_indexs
            ]
            updated_cate_list["collection_name"] = rearranged_collection_names

            if not survey_id or not updated_cate_list:
                return JsonResponse({"error": "Missing survey_id or updated_cate_list"}, status=400)

            save_dir = os.path.join('./src/static/data/info/', str(survey_id))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'cluster_info_updated.json')

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(updated_cate_list, f, ensure_ascii=False, indent=4)

            return JsonResponse({"message": "Cluster info updated and saved successfully!"}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)

import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

@csrf_exempt
def save_outline(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            updated_outline = data.get('outline', [])

            outline_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Finish the outline of the survey paper..."
                    },
                    {
                        "role": "user",
                        "content": "Finish the outline..."
                    }
                ],
                "outline": str(updated_outline)
            }

            file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'txt', Global_survey_id,'outline.json')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(outline_data, file, indent=4, ensure_ascii=False)
            
            html = generateOutlineHTML_qwen(Global_survey_id)

            return JsonResponse({"status": "success", "html": html})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    else:
        return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

@csrf_exempt
def select_sections(request):
    sections = request.POST
    survey = {}

    for k,v in sections.items():
        survey['title'] = "A Survey of " + Survey_dict[Global_survey_id]

        if k == "abstract":
            survey['abstract'] = ["The issue of class imbalance is pervasive in various practical applications of machine learning and data mining, including information retrieval and filtering, and the detection of credit card fraud. The problem of imbalanced learning concerns the effectiveness of learning algorithms when faced with underrepresented data and severe class distribution skews. The classification of data with imbalanced class distribution significantly hinders the performance of most standard classifier learning algorithms that assume a relatively balanced class distribution and equal misclassification costs.",
                                  "In this survey, we present a comprehensive overview of predictive modeling on imbalanced data. We categorize existing literature into three clusters: Sampling approaches, Algorithmic approaches, and Meta-learning approaches, which we introduce in detail. Our aim is to provide readers with a thorough understanding of the different strategies proposed to tackle the class imbalance problem and evaluate their effectiveness in enhancing the performance of learning algorithms."]
        if k == "introduction":
            survey['introduction'] = [
              {
                'subtitle': 'Background',
                'content' : '''Class imbalance is a common problem in machine learning and data mining, where the distribution of classes in the training dataset is highly skewed, with one class being significantly underrepresented compared to the other(s). This issue is prevalent in many real-world applications, including fraud detection, medical diagnosis, anomaly detection, and spam filtering, to name a few.
                               \nThe problem of imbalanced data affects the performance of many learning algorithms, which typically assume a balanced class distribution and equal misclassification costs. When the data is imbalanced, standard learning algorithms tend to favor the majority class, resulting in low accuracy in predicting the minority class. This drawback can lead to serious consequences, such as false negative errors in fraud detection or misdiagnosis in medical applications.
                               \nTo address the class imbalance problem, various techniques have been proposed, including resampling methods, cost-sensitive learning, and ensemble methods, among others. Resampling methods involve creating synthetic samples or under/oversampling the minority/majority classes to balance the data. Cost-sensitive learning assigns different misclassification costs to different classes to prioritize the minority class's correct prediction. Ensemble methods combine multiple models to improve predictive performance.
                               \nThe effectiveness of these techniques varies depending on the dataset and problem at hand. Hence, it is crucial to conduct a comprehensive evaluation of the different approaches to identify the most suitable one for a specific application. As such, your survey paper aims to provide an overview of the current state-of-the-art predictive modeling techniques for imbalanced data and highlight their strengths and limitations.
                            '''
              },
             {
                'subtitle': 'Methodologies', # Sampling approaches, Algorithmic approaches, and Meta-learning approaches
                'content' : '''Exisiting works are mainly categorized into Sampling approaches, Algorithmic approaches, and Meta-learning approaches.
                              \nSampling approaches:
                              \nResampling techniques are among the most popular methods for handling imbalanced data. These techniques involve either oversampling the minority class or undersampling the majority class to create a more balanced dataset. Examples of oversampling methods include SMOTE (Synthetic Minority Over-sampling Technique), ADASYN (Adaptive Synthetic Sampling), and Borderline-SMOTE. Undersampling techniques include random undersampling and Tomek Links. Moreover, hybrid methods, which combine both oversampling and undersampling, have also been proposed.
                              \nAlgorithmic approaches:
                              \nAnother approach to address the class imbalance problem is to modify the learning algorithm itself. Examples of such algorithmic approaches include cost-sensitive learning, where different costs are assigned to different types of misclassifications. Another approach is to adjust the decision threshold of the classifier, where the threshold is shifted to increase sensitivity towards the minority class. Additionally, ensemble methods, such as bagging, boosting, and stacking, have been proposed to combine multiple classifiers to improve predictive performance.
                              \nMeta-learning approaches:
                              \nMeta-learning approaches aim to automatically select the most suitable sampling or algorithmic approach for a specific dataset and problem. These approaches involve training a meta-classifier on multiple base classifiers, each using a different sampling or algorithmic approach. The meta-classifier then selects the most appropriate approach based on the characteristics of the input dataset. Examples of meta-learning approaches include MetaCost, MetaCostNN, and RAkEL.
                              \nThese approaches have shown promising results in addressing the class imbalance problem. However, their effectiveness depends on the specific characteristics of the dataset and problem at hand. Therefore, a comprehensive evaluation of different approaches is necessary to identify the most suitable one for a particular application.
                            '''
             },
             {
                'subtitle': 'Reminder',
                'content' : 'The rest of the paper is organized as follows. In section 2, we introduce the class imbalance problem and its causes and characteristics. Evaluation metrics are addressed in section 3. Section 4 presents an overview of the existing techniques for handling imbalanced data. Applications is illustrated in Section 5. Section 6 shows challenges and open issues. Conclusion and future directions are in Section 7.'
             }
            ]

        if k == "c_and_c":
            survey['c_and_c'] = '''Imbalanced data is a common problem in many real-world applications of machine learning and data mining, where the distribution of classes is highly skewed, with one or more classes being significantly underrepresented compared to the others. This can occur due to various reasons, such as sampling bias, data collection limitations, class overlap, or natural class distribution. The causes of imbalanced data can differ across different domains and applications, and understanding them is essential for developing effective predictive modeling techniques.
                            \nIn addition to the causes, imbalanced data is characterized by several properties that make it challenging for traditional machine learning algorithms. Firstly, the data imbalance results in a class distribution bias, where the majority class dominates the data, and the minority class(es) are often overshadowed, leading to poor classification performance. Secondly, the imbalance can lead to an asymmetric misclassification cost, where misclassifying the minority class is often more costly than misclassifying the majority class, resulting in high false negative rates. Thirdly, imbalanced data can exhibit class overlap, where instances from different classes are difficult to distinguish, leading to low discriminative power of the features and classifiers. Finally, imbalanced data can pose challenges for model evaluation and comparison, as traditional performance metrics such as accuracy, precision, and recall, can be misleading or inadequate in imbalanced settings.
                            \nUnderstanding the causes and characteristics of imbalanced data is crucial for developing effective and efficient predictive modeling techniques that can handle such data. The next section of this survey will discuss the various approaches proposed in the literature to address the imbalanced learning problem, with a focus on sampling, algorithmic, and meta-learning approaches.
                            '''
        if k == "evaluation":
            survey['evaluation'] = '''Evaluation metrics are an essential aspect of machine learning and data mining, as they quantify the performance of predictive models on a given dataset. In the case of imbalanced data, traditional evaluation metrics such as accuracy, precision, and recall may not be sufficient or even appropriate due to the class imbalance and asymmetry in misclassification costs. Therefore, alternative metrics have been proposed to measure the performance of predictive models on imbalanced datasets.
                            \nOne commonly used evaluation metric for imbalanced data is the area under the receiver operating characteristic curve (AUC-ROC). The AUC-ROC is a measure of the model's ability to distinguish between positive and negative instances and is computed as the area under the curve of the ROC plot. The ROC plot is a graphical representation of the trade-off between true positive rate (TPR) and false positive rate (FPR) for different decision thresholds. A perfect classifier would have an AUC-ROC score of 1, while a random classifier would have a score of 0.5.
                            \nAnother popular evaluation metric for imbalanced data is the area under the precision-recall curve (AUC-PR). The AUC-PR measures the precision-recall trade-off of the model and is computed as the area under the curve of the precision-recall plot. The precision-recall plot shows the relationship between precision and recall for different decision thresholds. A perfect classifier would have an AUC-PR score of 1, while a random classifier would have a score proportional to the ratio of positive to negative instances.
                            \nOther evaluation metrics for imbalanced data include F-measure, geometric mean, balanced accuracy, and cost-sensitive measures such as weighted and cost-sensitive versions of traditional metrics. F-measure is a harmonic mean of precision and recall, which balances the trade-off between them. The geometric mean is another metric that balances TPR and FPR and is useful in highly imbalanced datasets. Balanced accuracy is the average of TPR and TNR (true negative rate) and is useful in datasets where the class imbalance is extreme. Cost-sensitive measures incorporate the cost of misclassification and can be tailored to the specific application domain.
                            \nChoosing an appropriate evaluation metric for imbalanced data is essential to avoid biased or misleading performance estimates. The selection of metrics should be based on the application requirements, the class distribution, and the misclassification costs. In the next section, we will discuss various sampling, algorithmic, and meta-learning approaches proposed in the literature to address the imbalanced learning problem and their associated evaluation metrics.
                            '''

        if k == "methodology":
            survey['methodology'] = [
                'Our survey categorized existing works into three types: Sampling approaches, Algorithmic approaches, and Meta-learning approaches. Sampling approaches involve oversampling or undersampling, while algorithmic approaches modify the learning algorithm itself. Meta-learning approaches aim to automatically select the most suitable approach based on the characteristics of the input dataset.',
                [{'subtitle': 'Sampling approaches',
                  'content': 'For sampling approaches, Batista, et al. [1] proposed a simple experimental design to assess the performance of class imbalance treatment methods.  E.A.P.A. et al. [2] performs a broad experimental evaluation involving ten methods, three of them proposed by the authors, to deal with the class imbalance problem in thirteen uci data sets.  Batuwita, et al. [3] presents a method to improve fsvms for cil (called fsvm-cil), which can be used to handle the class imbalance problem in the presence of outliers and noise.  V. et al. [4] implements a wrapper approach that computes the amount of under-sampling and synthetic generation of the minority class examples (smote) to improve minority class accuracy.  Chen, et al. [5] presents ranked minority oversampling in boosting (ramoboost), which is a ramo technique based on the idea of adaptive synthetic data generation in an ensemble learning system.  Chen, et al. [6] proposes a new feature selection method, feature assessment by sliding thresholds (fast), which is based on the area under a roc curve generated by moving the decision boundary of a single feature classifier with thresholds placed using an even-bin distribution.  Davis, et al. [7] shows that a deep connection exists between roc space and pr space, such that a curve dominates in roc space if and only if it dominates in pr space.  In classifying documents, the system combines the predictions of the learners by applying evolutionary techniques as well [8]. Ertekin, et al. [9] is concerns with the class imbalance problem which has been known to hinder the learning performance of classification algorithms.  Ertekin, et al. [10] demonstrates that active learning is capable of solving the problem.  Garcı́aÿ, et al. [11] analyzes a generalization of a new metric to evaluate the classification performance in imbalanced domains, combining some estimate of the overall accuracy with a plain index about how dominant the class with the highest individual accuracy is.  Ghasemi, et al. [12] proposes an active learning algorithm that can work when only samples of one class as well as a set of unlabeled data are available.  He, et al. [13] provides a comprehensive review of the development of research in learning from imbalanced data.  Li, et al. [14] proposes an oversampling method based on support degree in order to guide people to select minority class samples and generate new minority class samples.  Li, et al. [15] analyzes the intrinsic factors behind this failure and proposes a suitable re-sampling method.  Liu, et al. [16] proposes two algorithms to overcome this deficiency.  J. et al. [17] considers the application of these ensembles to imbalanced data : classification problems where the class proportions are significantly different.  Seiffert, et al. [18] presents a new hybrid sampling/boosting algorithm, called rusboost, for learning from skewed training data.  Song, et al. [19] proposes an improved adaboost algorithm called baboost (balanced adaboost), which gives higher weights to the misclassified examples from the minority class.  Sun, et al. [20] develops a cost-sensitive boosting algorithm to improve the classification performance of imbalanced data involving multiple classes.  Van et al. [21] presents a comprehensive suite of experimentation on the subject of learning from imbalanced data.  Wasikowski, et al. [22] presents a first systematic comparison of the three types of methods developed for imbalanced data classification problems and of seven feature selection metrics evaluated on small sample data sets from different applications.  an active under-sampling approach is proposed for handling the imbalanced problem in Yang, et al. [23]. Zhou, et al. [24] studies empirically the effect of sampling and threshold-moving in training cost-sensitive neural networks. \n'},
                 {'subtitle': 'Algorithmic approaches',
                  'content': 'For algorithmic approaches, Baccianella, et al. [25] proposed a simple way to turn standard measures for or into ones robust to imbalance.  Lin, et al. [26] applies a fuzzy membership to each input point and reformulate the svms such that different input points can make different constributions to the learning of decision surface. \n'},
                 {'subtitle': 'Meta-learning approaches',
                  'content': 'For meta-learning approaches, Drummond, et al. [27] proposed an alternative to roc representation, in which the expected cost of a classi er is represented explicitly.  Tao, et al. [28] develops a mechanism to overcome these problems.  Torgo et al. [29] presents a generalization of regression error characteristic (rec) curves.  C. et al. [30] demonstrates that class probability estimates attained via supervised learning in imbalanced scenarios systematically underestimate the probabilities for minority class instances, despite ostensibly good overall calibration.  Yoon, et al. [31] proposes preprocessing majority instances by partitioning them into clusters.  Zheng, et al. [32] investigates the usefulness of explicit control of that combination within a proposed feature selection framework.'}]]



        if k == "app":
            survey['app'] = '''The problem of imbalanced data is pervasive in many real-world applications of predictive modeling, where the data is often skewed towards one or more minority class or classes. Such applications include, but are not limited to, fraud detection in finance, rare disease diagnosis in healthcare, fault detection in manufacturing, spam filtering in email systems, and anomaly detection in cybersecurity. In these scenarios, accurately identifying the minority class instances is of utmost importance, as they often represent critical and rare events that have significant impact or consequences.
                            \nHowever, traditional classification algorithms tend to perform poorly on imbalanced datasets, since they are often biased towards the majority class due to its abundance in the data. This results in low accuracy, high false negative rates, and poor generalization performance, especially for the minority class(es) of interest. In addition, the cost of misclassifying the minority class is often much higher than that of the majority class, making it even more critical to achieve high accuracy and low false negative rates for these instances.
                            \nTo overcome the class imbalance problem, a variety of predictive modeling techniques have been proposed and developed in the literature, specifically designed to handle imbalanced datasets. These techniques range from simple preprocessing methods that adjust the class distribution, to more complex algorithmic modifications that incorporate class imbalance considerations into the learning process. The effectiveness of these techniques depends on the specific characteristics of the dataset and problem, and thus, their selection and evaluation require careful experimentation and analysis.
                            \nOverall, the development and application of predictive modeling techniques for imbalanced data is an active and important research area, with many practical and societal implications. Advancements in this field have the potential to improve the accuracy, efficiency, and fairness of many critical applications, and thus, benefit society as a whole.
                            '''

        if k == "app":
            survey['clg'] = '''Selecting the most appropriate sampling, algorithmic, or meta-learning approach for a specific dataset: There is no one-size-fits-all solution, and choosing the right approach can be challenging.
                            \nLack of standard evaluation metrics that can capture the performance of models on imbalanced data, especially for rare events: Existing evaluation metrics like accuracy can be misleading in imbalanced datasets, and there is a need for metrics that can capture the performance of models on rare events.
                            \nInterpretability and explainability of models trained on imbalanced data: It can be difficult to understand how a model arrives at its predictions, especially when the data is heavily skewed, and there is a need for more interpretable models.
                            \nScalability of methods to handle very large datasets with imbalanced class distributions: As datasets grow in size, it can be challenging to scale methods to handle the imbalanced class distribution efficiently.
                            \nNeed for better feature engineering techniques to handle imbalanced data: Feature engineering is an important step in predictive modeling, and there is a need for better techniques that can handle imbalanced data.
                            \nDevelopment of new learning algorithms that are specifically designed to work well on imbalanced datasets: Most standard learning algorithms assume a relatively balanced class distribution, and there is a need for new algorithms that can handle imbalanced data more effectively.
                            \nResearch into the use of semi-supervised and unsupervised learning techniques for imbalanced data: Semi-supervised and unsupervised learning techniques have shown promise in imbalanced data, and there is a need for more research to explore their potential.
                            \nPotential benefits of using ensemble methods to combine multiple models trained on imbalanced data: Ensemble methods can improve the performance of models on imbalanced data by combining multiple models, and there is a need for more research to explore their potential.
                            \nDeveloping more effective methods for dealing with concept drift and evolving class distributions over time in imbalanced datasets: As class distributions evolve over time, it can be challenging to adapt models to the new distribution, and there is a need for more effective methods to handle concept drift.
                            '''


        if k == "conclusion":
            conclusion = '''In conclusion, the class imbalance problem is a significant challenge in predictive modeling, which can lead to biased models and poor performance. In this survey, we have provided a comprehensive overview of existing works on predictive modeling on imbalanced data. We have discussed different approaches to address this problem, including sampling approaches, algorithmic approaches, and meta-learning approaches, as well as evaluation metrics and challenges in this field. We also presented some potential future research directions in this area. The insights and knowledge provided in this survey paper can help researchers and practitioners better understand the challenges and opportunities in predictive modeling on imbalanced data and design more effective approaches to address this problem in real-world applications.
            \nThere are also some potencial directions for future research:
            \n1. Incorporating domain knowledge: Incorporating domain-specific knowledge can help improve the performance of models on imbalanced data. Research can be done on developing techniques to effectively integrate domain knowledge into the modeling process.
            \n2. Explainability of models: With the increasing adoption of machine learning models in critical applications, it is important to understand how the models make predictions. Research can be done on developing explainable models for imbalanced data, which can provide insights into the reasons for model predictions.
            \n3. Online learning: Imbalanced data can evolve over time, and models need to be adapted to new data as it becomes available. Research can be done on developing online learning algorithms that can adapt to imbalanced data in real-time.
            \n4. Multi-label imbalanced classification: In many real-world scenarios, multiple classes can be imbalanced simultaneously. Research can be done on developing techniques for multi-label imbalanced classification that can effectively handle such scenarios.
            \n5. Transfer learning: In some cases, imbalanced data in one domain can be used to improve the performance of models in another domain. Research can be done on developing transfer learning techniques for imbalanced data, which can leverage knowledge from related domains to improve performance.
            \n6. Incorporating fairness considerations: Models trained on imbalanced data can have biases that can disproportionately affect certain groups. Research can be done on developing techniques to ensure that models trained on imbalanced data are fair and do not discriminate against any particular group.
            \n7. Imbalanced data in deep learning: Deep learning has shown great promise in various applications, but its effectiveness on imbalanced data is not well understood. Research can be done on developing techniques to effectively apply deep learning to imbalanced data.
            \n8. Large-scale imbalanced data: With the increasing availability of large-scale datasets, research can be done on developing scalable techniques for predictive modeling on imbalanced data.
            '''
            survey['conclusion'] = conclusion

    survey['references'] = []
    try:
        for ref in Global_df_selected['ref_entry']:
            entry = str(ref)
            survey['references'].append(entry)
    except:
        import traceback
        print(traceback.print_exc())

    survey_dict = json.dumps(survey)

    return HttpResponse(survey_dict)

@csrf_exempt
def get_survey(request):
    survey_dict = get_survey_text()
    survey_dict = json.dumps(survey_dict)
    return HttpResponse(survey_dict)
    
@csrf_exempt
@timeout_handler(1800)  # 30分钟超时
def get_survey_id_sync(request):
    """同步版本的获取调研ID函数"""
    start_time = time.time()
    operation_id = f"survey_{int(start_time)}"
    update_progress(operation_id, 0, "Starting survey generation...")
    
    global Global_survey_id, Global_survey_title, Global_collection_names_clustered, Global_citation_data
    
    try:
        update_progress(operation_id, 10, "Initializing survey generation...")
        
        if not Global_survey_id:
            update_progress(operation_id, -1, "Survey ID not found")
            return JsonResponse({"error": "Survey ID not found"}, status=400)
        
        if not Global_collection_names_clustered:
            update_progress(operation_id, -1, "No clustered collections found")
            return JsonResponse({"error": "No clustered collections found"}, status=400)
        
        update_progress(operation_id, 20, "Preparing survey data...")
        
        print("Global_collection_names_clustered: ")
        for i, element in enumerate(Global_collection_names_clustered):
            print(f"第 {i} 个元素：{element}")
        
        update_progress(operation_id, 30, "Generating survey content...")
        
        # 在子线程中执行survey生成，以便能够跟踪进度
        def generate_survey_with_progress():
            try:
                update_progress(operation_id, 40, "Generating survey outline...")
                
                # 这里调用实际的survey生成函数，不再需要pipeline参数
                generateSurvey_qwen_new(
                    Global_survey_id, 
                    Global_survey_title, 
                    Global_collection_names_clustered, 
                    None,  # pipeline参数设置为None，函数内部已经改为API调用
                    Global_citation_data,
                    embedder = get_embedder()
                )
                
                update_progress(operation_id, 90, "Survey generation completed!")
                return True
                
            except Exception as e:
                update_progress(operation_id, -1, f"Survey generation failed: {str(e)}")
                print(f"Error in generateSurvey_qwen_new: {e}")
                return False
        
        # 执行survey生成
        success = generate_survey_with_progress()
        
        if success:
            update_progress(operation_id, 100, "Survey ready!")
            
            response_data = {
                "survey_id": Global_survey_id,
                "message": "Survey generated successfully",
                "operation_id": operation_id,
                "processing_time": round(time.time() - start_time, 2)
            }
            
            return JsonResponse(response_data)
        else:
            return JsonResponse({"error": "Survey generation failed"}, status=500)
            
    except TimeoutError as e:
        update_progress(operation_id, -1, f"Survey generation timed out: {str(e)}")
        return JsonResponse({'error': f'Survey generation timed out after 30 minutes: {str(e)}'}, status=408)
    except Exception as e:
        update_progress(operation_id, -1, f"Survey generation failed: {str(e)}")
        return JsonResponse({'error': f'Survey generation failed: {str(e)}'}, status=500)

@csrf_exempt
@timeout_handler(900)  # 15分钟超时
def generate_pdf_sync(request):
    if request.method == 'POST':
        # 获取operation_id用于进度跟踪
        operation_id = getattr(request, 'operation_id', f"pdf_{int(time.time())}")
        update_progress(operation_id, 10, "Starting PDF generation...")
        
        survey_id = request.POST.get('survey_id', '') or Global_survey_id
        if not survey_id:
            update_progress(operation_id, -1, "Missing survey_id; cannot generate PDF filename")
            return JsonResponse({'error': 'survey_id is required (and no global survey ID is set).'}, status=400)
        markdown_content = request.POST.get('content', '')
        
        update_progress(operation_id, 20, "Processing markdown content...")
        
        markdown_dir = f'./src/static/data/info/{survey_id}/'
        markdown_filename = f'survey_{survey_id}_vanilla.md'
        markdown_filepath = os.path.join(markdown_dir, markdown_filename)

        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir)
            print(f"Directory '{markdown_dir}' created.")
        else:
            print(f"Directory '{markdown_dir}' already exists.")

        with open(markdown_filepath, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(markdown_content)
        print(f"Markdown content saved to: {markdown_filepath}")

        update_progress(operation_id, 40, "Finalizing survey paper...")
        
        markdown_content = finalize_survey_paper(markdown_content, Global_collection_names, Global_file_names)
        markdown_dir = f'./src/static/data/info/{survey_id}/'
        markdown_filename = f'survey_{survey_id}_processed.md'
        markdown_filepath = os.path.join(markdown_dir, markdown_filename)

        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir)
            print(f"Directory '{markdown_dir}' created.")
        else:
            print(f"Directory '{markdown_dir}' already exists.")

        with open(markdown_filepath, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(markdown_content)
        print(f"Markdown content saved to: {markdown_filepath}")

        update_progress(operation_id, 60, "Generating PDF file...")

        pdf_filename = f'survey_{survey_id}.pdf'
        pdf_dir = './src/static/data/results'
        pdf_filepath = os.path.join(pdf_dir, pdf_filename)

        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            print(f"Directory '{pdf_dir}' created.")
        else:
            print(f"Directory '{pdf_dir}' already exists.")

        print(f"PDF will be saved to: {pdf_filepath}")

        update_progress(operation_id, 80, "Converting markdown to PDF...")

        pdf = MarkdownPdf()
        pdf.meta["title"] = "Survey Results"
        pdf.add_section(Section(markdown_content, toc=False))
        pdf.save(pdf_filepath)

        update_progress(operation_id, 100, "PDF generation completed!")

        # 返回JSON格式的结果而不是二进制PDF数据
        return JsonResponse({
            'success': True,
            'message': 'PDF generated successfully',
            'survey_id': survey_id,
            'pdf_filename': pdf_filename,
            'pdf_path': pdf_filepath
        })

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def generate_pdf(request):
    """异步版本的PDF生成接口，避免Cloudflare 524超时"""
    if request.method == 'POST':
        operation_id = f"pdf_{int(time.time())}"
        success = task_manager.start_task(
            operation_id,
            generate_pdf_sync,
            request
        )
        if not success:
            return JsonResponse({'error': 'PDF generation task already running'}, status=409)
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'PDF generation started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
@timeout_handler(900)  # 15分钟超时
def generate_pdf_from_tex_sync(request):
    global Global_survey_id, Global_survey_title
    if request.method == 'POST':
        # 获取operation_id用于进度跟踪
        operation_id = getattr(request, 'operation_id', f"latex_{int(time.time())}")
        update_progress(operation_id, 10, "Starting LaTeX PDF generation...")
        
        print(f"Request content type: {request.content_type}")
        print(f"Request POST data: {request.POST}")
        print(f"Request FILES: {request.FILES}")
        survey_id = request.POST.get('survey_id', '') or Global_survey_id
        markdown_content = request.POST.get('content', '')
        print(f"Survey ID: {survey_id}")
        print(f"Has content: {bool(markdown_content)}")
        
        if not survey_id:
            return JsonResponse({'error': 'survey_id is missing'}, status=400)
            
        update_progress(operation_id, 20, "Setting up directories...")
        
        base_dir = f'./src/static/data/info/{survey_id}'
        md_path = os.path.join(base_dir, f'survey_{survey_id}_processed.md')
        new_md_path = os.path.join(base_dir, f'survey_{survey_id}_preprocessed.md')
        tex_path = os.path.join(base_dir, 'template.tex')
        new_tex_path = os.path.join(base_dir, 'template_with_figure.tex')
        sty_path = os.path.join(base_dir, 'acl.sty')
        pdf_dir = './src/static/data/results'
        os.makedirs(base_dir, exist_ok=True)
        print(f"Directory '{base_dir}' checked or created.")
        
        update_progress(operation_id, 30, "Copying template files...")
        
        origin_template = 'src/demo/latex_template/template.tex'
        origin_acl_sty = 'src/demo/latex_template/acl.sty'
        shutil.copy(origin_template, tex_path)
        shutil.copy(origin_acl_sty, sty_path)
        os.makedirs(pdf_dir, exist_ok=True)
        
        update_progress(operation_id, 40, "Processing survey content...")
        
        # 如果传入了content且processed.md文件不存在，则创建它
        if markdown_content and not os.path.exists(md_path):
            # 先保存原始内容
            vanilla_md_path = os.path.join(base_dir, f'survey_{survey_id}_vanilla.md')
            with open(vanilla_md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"Vanilla markdown saved to: {vanilla_md_path}")
            
            # 处理并保存最终的markdown
            processed_content = finalize_survey_paper(markdown_content, Global_collection_names, Global_file_names)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            print(f"Processed markdown saved to: {md_path}")
        elif not os.path.exists(md_path):
            return JsonResponse({'error': f'Processed markdown file not found: {md_path}. Please generate regular PDF first or provide survey content.'}, status=400)
        
        update_progress(operation_id, 50, "Processing markdown content...")
        
        preprocess_md(md_path, new_md_path)
        md_to_tex(new_md_path, tex_path, Global_survey_title)
        
        update_progress(operation_id, 70, "Inserting figures and formatting...")
        
        insert_figures(
            png_path=f'src/static/data/info/{survey_id}/outline.png',
            tex_path= tex_path, 
            json_path=f'src/static/data/info/{survey_id}/flowchart_results.json',
            ref_names= Global_ref_list,
            survey_title=Global_survey_title,
            new_tex_path=new_tex_path
        )
        
        update_progress(operation_id, 85, "Compiling LaTeX to PDF...")
        
        tex_to_pdf(
            new_tex_path,
            output_dir=os.path.dirname(new_tex_path),
            compiler="xelatex"
        )
        pdf_path = os.path.join(os.path.dirname(new_tex_path), 'template_with_figure.pdf' )
        final_pdf_path = os.path.join(pdf_dir, f'survey_{survey_id}_latex.pdf')
        shutil.copy2(pdf_path, final_pdf_path)
        
        update_progress(operation_id, 100, "LaTeX PDF generation completed!")
        
        # 返回JSON格式的结果而不是二进制PDF数据
        return JsonResponse({
            'success': True,
            'message': 'LaTeX PDF generated successfully',
            'survey_id': survey_id,
            'pdf_filename': f'survey_{survey_id}_latex.pdf',
            'pdf_path': final_pdf_path
        })
        
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def get_refs(topic):
    '''
    Get the references from given topic
    Return with a list
    '''
    default_references = ['ref1','ref2','ref3','ref4','ref5','ref6','ref7','ref8','ref9','ref10']
    default_ref_links = ['', '', '', '', '', '', '', '', '', '']
    default_ref_ids = ['', '', '', '', '', '', '', '', '', '']
    references = []
    ref_links = []
    ref_ids = []

    try:
        ## here is the algorithm part
        ref_path   = os.path.join(DATA_PATH, topic + '.tsv')
        df         = pd.read_csv(ref_path, sep='\t')
        for i,r in df.iterrows():
            # print(r['intro'], r['ref_title'], i)
            if not pd.isnull(r['intro']):
                references.append(r['ref_title'])
                ref_links.append(r['ref_link'])
                ref_ids.append(i)
    except:
        print(traceback.print_exc())
        references = default_references
        ref_links = default_ref_links
        ref_ids = default_ref_ids
    print(len(ref_ids))
    return references, ref_links, ref_ids

def get_survey_text(refs=Global_ref_list):
    '''
    Get the survey text from a given ref list
    Return with a dict as below default value:
    '''
    survey = {
        'Title': "A Survey of " + Survey_dict[Global_survey_id],
        'Abstract': "test "*150,
        'Introduction': "test "*500,
        'Methodology': [
            "This is the proceeding",
            [{"subtitle": "This is the first subtitle", "content": "test "*500},
             {"subtitle": "This is the second subtitle", "content": "test "*500},
             {"subtitle": "This is the third subtitle", "content": "test "*500}]
        ],
        'Conclusion': "test "*150,
        'References': []
    }

    try:
        ## abs generation
        abs, last_sent = absGen(Global_survey_id, Global_df_selected, Global_category_label)
        survey['Abstract'] = [abs, last_sent]

        ## Intro generation
        #intro = introGen_supervised(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        intro = introGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Introduction'] = intro

        ## Methodology generation
        proceeding, detailed_des = methodologyGen(Global_survey_id, Global_df_selected, Global_category_label, Global_category_description)
        survey['Methodology'] = [proceeding, detailed_des]

        ## Conclusion generation
        conclusion = conclusionGen(Global_survey_id, Global_category_label)
        survey['Conclusion'] = conclusion

        try:
            for ref in Global_df_selected['ref_entry']:
                entry = str(ref)
                survey['References'].append(entry)
        except:
            colors, category_label, category_description = Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
            for ref in Global_df_selected['ref_entry']:
                entry = str(ref)
                survey['References'].append(entry)

    except:
        print(traceback.print_exc())
    return survey

def Clustering_refs(n_clusters):
    global Global_cluster_num
    df = pd.read_csv(TSV_PATH + Global_survey_id + '.tsv', sep='\t', index_col=0, encoding='utf-8')

    print(Global_ref_list)
    df_selected = df.iloc[Global_ref_list]
    df_selected, colors, best_n_topics = clustering(df_selected, [3,4,5], Global_survey_id)
    Global_cluster_num = best_n_topics

    global Global_df_selected
    Global_df_selected = df_selected
    category_description = [0]*len(colors)
    category_label = [0]*len(colors)

    return colors, category_label
    # return 1,0,1

def remove_invalid_citations(text, valid_collection_names):
    pattern = r"\[(.*?)\\\]"
    all_matches = re.findall(pattern, text)

    new_text = text
    for match in all_matches:
        cleaned_match = match.rstrip('\\')
        if cleaned_match not in valid_collection_names:
            new_text = new_text.replace(f"[{match}\\]", "")
    return new_text

# wza
def normalize_citations_with_mapping(paper_text):
    citations = re.findall(r'\[.*?\]', paper_text)
    unique_citations = list(dict.fromkeys(citations))
    citation_mapping = {citation: f'[{i + 1}]' for i, citation in enumerate(unique_citations)}

    normalized_text = paper_text
    for old_citation, new_citation in citation_mapping.items():
        normalized_text = normalized_text.replace(old_citation, new_citation)

    reverse_mapping = {
        i + 1: unique_citations[i].strip('[]').rstrip('\\')
        for i in range(len(unique_citations))
    }

    return normalized_text, reverse_mapping

def generate_references_section(citation_mapping, collection_pdf_mapping):
    references = ["# References"]
    ref_list = []
    for num in sorted(citation_mapping.keys()):
        collection_name = citation_mapping[num]
        pdf_name = collection_pdf_mapping.get(collection_name, "Unknown PDF")
        if pdf_name.endswith(".pdf"):
            pdf_name = pdf_name[:-4]
        ref_list.append(pdf_name)
        # 在每一行末尾添加两个空格以确保换行
        references.append(f"[{num}] {pdf_name}  ")

    return "\n".join(references), ref_list

def fix_citation_punctuation_md(text):
    pattern = r'\.\s*(\\\[\d+\])'
    replacement = r' \1.'
    fixed_text = re.sub(pattern, replacement, text)
    return fixed_text

def finalize_survey_paper(paper_text, 
                          Global_collection_names, 
                          Global_file_names):
    global Global_survey_id, Global_survey_title, Global_ref_list

    paper_text = remove_invalid_citations(paper_text, Global_collection_names)
    normalized_text, citation_mapping = normalize_citations_with_mapping(paper_text)
    normalized_text = fix_citation_punctuation_md(normalized_text)
    collection_pdf_mapping = dict(zip(Global_collection_names, Global_file_names))
    
    references_section, ref_list = generate_references_section(citation_mapping, collection_pdf_mapping)
    Global_ref_list = ref_list
    print(ref_list)

    json_path = os.path.join("src", "static", "data", "txt", Global_survey_id, "outline.json")
    output_png_path = os.path.join("src", "static", "data", "info", Global_survey_id, "outline")
    md_path = os.path.join("src", "static", "data", "info", Global_survey_id, f"survey_{Global_survey_id}_processed.md")
    flowchart_results_path = os.path.join("src", "static", "data", "info", Global_survey_id, "flowchart_results.json")
    detect_flowcharts(Global_survey_id)
    png_path = generate_graphviz_png(
        json_path=json_path,
        output_png_path=output_png_path,
        md_content=normalized_text,
        title=Global_survey_title,
        max_root_chars=30
    )

    try:
        normalized_text = insert_ref_images(flowchart_results_path, ref_list, normalized_text)
    except Exception as e:
        print(f"Error inserting ref image: {e}. Continuing with next step.")
    try:
        normalized_text = insert_outline_image(
            png_path=png_path,
            md_content=normalized_text,
            survey_title =Global_survey_title
        )
    except Exception as e:
        print(f"Error inserting outline image: {e}. Continuing with next step.")

    final_paper = normalized_text.strip() + "\n\n" + references_section
    return final_paper

# Cleanup function for Django shutdown
def cleanup_resources():
    """Clean up resources when Django shuts down"""
    try:
        cleanup_openai_client()
        cleanup_retriever()
        print("Successfully cleaned up resources")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup function for Django shutdown
import atexit
atexit.register(cleanup_resources)

@csrf_exempt  
def upload_refs(request):
    """异步版本的文件上传接口，立即返回operation_id避免Cloudflare 524超时。先保存文件到磁盘，再异步处理。"""
    if request.method == 'POST':
        operation_id = f"upload_{int(time.time())}"
        print(f"[DEBUG] Starting async upload task: {operation_id}")

        # 1. 先将所有上传文件保存到临时目录
        temp_dir = os.path.join('src', 'static', 'data', 'tmp_upload', operation_id)
        os.makedirs(temp_dir, exist_ok=True)
        file_paths = []
        for file_key, file in request.FILES.items():
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            file_paths.append(temp_path)

        # 2. 收集POST参数
        post_data = dict(request.POST.items())
        # 3. 启动异步任务，传递文件路径和参数
        def upload_refs_sync_wrapper(file_paths, post_data, operation_id):
            # 构造一个伪request对象，兼容原有upload_refs_sync逻辑
            class DummyRequest:
                method = 'POST'
                FILES = {}
                POST = post_data
            dummy_request = DummyRequest()
            dummy_request.file_paths = file_paths
            dummy_request.operation_id = operation_id
            return upload_refs_sync(dummy_request)

        success = task_manager.start_task(
            operation_id,
            upload_refs_sync_wrapper,
            file_paths,
            post_data,
            operation_id
        )
        if not success:
            print(f"[DEBUG] Task {operation_id} already running")
            return JsonResponse({'error': 'Upload task already running'}, status=409)
        print(f"[DEBUG] Async task {operation_id} started successfully")
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'File upload started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def automatic_taxonomy(request):
    """异步版本的自动分类接口，避免Cloudflare 524超时"""
    if request.method == 'POST':
        # 生成操作ID
        operation_id = f"taxonomy_{int(time.time())}"
        
        # 启动异步任务
        success = task_manager.start_task(
            operation_id, 
            automatic_taxonomy_sync, 
            request
        )
        
        if not success:
            return JsonResponse({'error': 'Taxonomy task already running'}, status=409)
        
        # 立即返回operation_id
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'Automatic taxonomy started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def get_survey_id(request):
    """异步版本的获取调研ID接口，避免Cloudflare 524超时"""
    if request.method == 'POST':
        # 生成操作ID
        operation_id = f"survey_{int(time.time())}"
        
        # 启动异步任务
        success = task_manager.start_task(
            operation_id, 
            get_survey_id_sync, 
            request
        )
        
        if not success:
            return JsonResponse({'error': 'Survey generation task already running'}, status=409)
        
        # 立即返回operation_id
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'Survey generation started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def generate_pdf_from_tex(request):
    """异步版本的LaTeX PDF生成接口，避免Cloudflare 524超时"""
    if request.method == 'POST':
        operation_id = f"latex_{int(time.time())}"
        success = task_manager.start_task(
            operation_id,
            generate_pdf_from_tex_sync,
            request
        )
        if not success:
            return JsonResponse({'error': 'LaTeX PDF generation task already running'}, status=409)
        return JsonResponse({
            'operation_id': operation_id,
            'status': 'started',
            'message': 'LaTeX PDF generation started successfully. Use the operation_id to check progress.',
            'progress_url': f'/get_operation_progress/?operation_id={operation_id}'
        })
    return JsonResponse({'error': 'Invalid request method'}, status=405)

# @csrf_exempt
# def test_async_simple(request):
#     """简单的异步测试函数，用于验证异步机制"""
#     if request.method == 'POST':
#         operation_id = f"test_{int(time.time())}"
#         
#         def simple_task(request):
#             """简单的测试任务"""
#             update_progress(operation_id, 10, "Starting test task...")
#             time.sleep(2)
#             update_progress(operation_id, 50, "Task half way...")
#             time.sleep(2)
#             update_progress(operation_id, 100, "Test task completed!")
#             return JsonResponse({'message': 'Test completed successfully', 'test_id': operation_id})
#         
#         success = task_manager.start_task(operation_id, simple_task, request)
#         
#         if not success:
#             return JsonResponse({'error': 'Test task already running'}, status=409)
#         
#         return JsonResponse({
#             'operation_id': operation_id,
#             'status': 'started',
#             'message': 'Test task started successfully.'
#         })
#     
#     return JsonResponse({'error': 'Invalid request method'}, status=405)