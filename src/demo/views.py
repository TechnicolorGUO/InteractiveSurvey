from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import os
import json
import requests
import time
import pandas as pd
import numpy as np
import shutil
import traceback

from demo.ref_paper_desp import ref_desp
import hashlib
import pdb
import re
import pke
import networkx as nx
from collections import defaultdict
import os
import csv

from django.http import JsonResponse
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.utils.text import slugify
from django.middleware.csrf import get_token

from .asg_loader import DocumentLoading
# from .parse import DocumentLoading
from .asg_retriever import legal_pdf, process_pdf, query_embeddings,query_embeddings_new, query_embeddings_new_new, query_multiple_collections
from .asg_generator import generate,generate_sentence_patterns
from .asg_outline import OutlineGenerator, generateOutlineHTML,generateOutlineHTML_qwen, generateSurvey,generateSurvey_qwen, generateSurvey_qwen_new
from .asg_clustername import generate_cluster_name_qwen_sep, refine_cluster_name, generate_cluster_name_new
from .postprocess import reindex_citations, generate_references_section
import glob
import nltk

from langchain_huggingface import HuggingFaceEmbeddings
import transformers
import torch
from dotenv import load_dotenv

load_dotenv()

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
Global_test_flag = True
Global_collection_names = []
Global_collection_names_clustered = []
Global_file_names=[]
Global_description_list = []
Global_pipeline = None
Global_cluster_names = []
Global_citation_data = []


embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


from demo.taskDes import absGen, introGen,introGen_supervised, methodologyGen, conclusionGen
from demo.category_and_tsne import clustering


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
Global_pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    token = os.getenv('HF_API_KEY'),
    device_map="auto",
)
Global_pipeline.model.load_adapter(peft_model_id = "technicolor/llama3.1_8b_outline_generation", adapter_name="outline")
Global_pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_abstract_generation", adapter_name="abstract")
Global_pipeline.model.load_adapter(peft_model_id ="technicolor/llama3.1_8b_conclusion_generation", adapter_name="conclusion")

# Global_pipeline.model.disable_adapters()

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
        #print(matched_entries.shape)
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

class PosRank(pke.unsupervised.PositionRank):
    def __init__(self):
        """Redefining initializer for PositionRank."""
        super(PosRank, self).__init__()
        self.positions = defaultdict(float)
        """Container the sums of word's inverse positions."""
    def candidate_selection(self,grammar=None,maximum_word_number=3,minimum_word_number=2):
        if grammar is None:
            grammar = "NP:{<ADJ>*<NOUN|PROPN>+}"

        # select sequence of adjectives and nouns
        self.grammar_selection(grammar=grammar)

        # filter candidates greater than 3 words
        for k in list(self.candidates):
            v = self.candidates[k]
            #pdb.set_trace()
            #if len(k) < 3:
            #    del self.candidates[k]
            if len(v.lexical_form) > maximum_word_number or len(v.lexical_form) < minimum_word_number:
                #if len(v.lexical_form) < minimum_word_number:
                #    pdb.set_trace()
                del self.candidates[k]

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
    #extractor = pke.unsupervised.TextRank()
    #extractor = pke.unsupervised.PositionRank()
    extractor = PosRank()

    #input_str=input_pd["abstract"][0].replace('-','')#.value()

    #pdb.set_trace()

    #for (keyphrase, score) in extractor.get_n_best(n=5, stemming=True):#stemming=False
    #    print(keyphrase, score)
    abs_top5_unigram_list_list = []
    abs_top5_bigram_list_list = []
    abs_top5_trigram_list_list = []
    intro_top5_unigram_list_list = []
    intro_top5_bigram_list_list = []
    intro_top5_trigram_list_list = []

    for line_index,pd_row in input_pd.iterrows():

        input_str=pd_row["abstract"].replace('-','')
        extractor.load_document(input=input_str,language='en',normalization=None)
        #extractor.load_document(input=input_str,language="en",normalization='stemming')

        #unigram
        unigram_extractor=extractor
        #unigram_extractor.candidate_weighting(window=1,pos=pos,top_percent=0.33)
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
        #bigram_extractor.candidate_weighting(window=2,pos=pos,top_percent=0.33)
        #abs_top5_bigram = extractor.get_n_best(n=5, stemming=True)#stemming=False
        bigram_extractor.candidate_selection(maximum_word_number=2,minimum_word_number=2)
        bigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_bigram_list = []
        for (keyphrase, score) in bigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_bigram_list.append(keyphrase)

        #trigram
        trigram_extractor=extractor
        #trigram_extractor.candidate_weighting(window=3,pos=pos,top_percent=0.33)
        trigram_extractor.candidate_selection(maximum_word_number=3,minimum_word_number=3)
        trigram_extractor.candidate_weighting(window=6,pos=pos,normalized=False)
        abs_top5_trigram_list = []
        for (keyphrase, score) in trigram_extractor.get_n_best(n=5, stemming=True):
            keyphrase = keyphrase.replace('-','')
            if len(keyphrase)>2:
                abs_top5_trigram_list.append(keyphrase)

        '''
        input_str=pd_row["intro"].replace('-','')
        extractor.load_document(input=input_str,language='en',normalization=None)

        #unigram
        extractor.candidate_weighting(window=1,pos=pos,top_percent=0.33)
        intro_top5_unigram_list = []
        for (keyphrase, score) in extractor.get_n_best(n=5, stemming=True):
            intro_top5_unigram_list.append(keyphrase)

        #bigram
        extractor.candidate_weighting(window=2,pos=pos,top_percent=0.33)
        #intro_top5_bigram = extractor.get_n_best(n=5, stemming=True)#stemming=False
        intro_top5_bigram_list = []
        for (keyphrase, score) in extractor.get_n_best(n=5, stemming=True):
            intro_top5_bigram_list.append(keyphrase)

        #trigram
        extractor.candidate_weighting(window=3,pos=pos,top_percent=0.33)
        intro_top5_trigram_list = []
        for (keyphrase, score) in extractor.get_n_best(n=5, stemming=True):
            intro_top5_trigram_list.append(keyphrase)
        '''

        abs_top5_unigram_list_list.append(abs_top5_unigram_list)
        abs_top5_bigram_list_list.append(abs_top5_bigram_list)
        abs_top5_trigram_list_list.append(abs_top5_trigram_list)
        '''
        intro_top5_unigram_list_list.append(intro_top5_unigram_list)
        intro_top5_bigram_list_list.append(intro_top5_bigram_list)
        intro_top5_trigram_list_list.append(intro_top5_trigram_list)
        '''
    return abs_top5_unigram_list_list,abs_top5_bigram_list_list,abs_top5_trigram_list_list

def process_file(file_name, survey_id):
    # parser = DocumentLoading()
    global embedder
    result = process_pdf(file_name, survey_id, embedder)
    collection_name = result[0]
    name = result[-1]
    print(name)
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    return collection_name, name

def sanitize_filename_py(filename):
    last_dot = filename.rfind('.')
    
    def sanitize_part(part):
        # Convert to lowercase
        part = part.lower()
        # Replace non-alphanumerics with space
        part = re.sub(r'[^a-z0-9]', ' ', part)
        # Collapse multiple spaces
        part = re.sub(r'\s+', ' ', part)
        # Strip leading/trailing spaces
        part = part.strip()
        
        # Split into words
        words = part.split(' ')
        
        if len(words) == 0:
            return ''
        
        # Capitalize the first word
        words[0] = words[0].capitalize()
        
        # Rejoin with spaces
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

@csrf_exempt
def upload_refs(request):
    start_time = time.time()
    if request.method == 'POST':
        if not request.FILES:
            return JsonResponse({'error': 'No file part'}, status=400)

        is_valid_submission = True
        has_label_id = False
        has_ref_link = False

        filenames = []
        collection_names = []
        filesizes = []
        file_dict = request.FILES
        file_name = list(file_dict.keys())[0]
        print(file_dict)
        print(list(file_dict.keys()))

        global Global_survey_id
        global Global_test_flag
        global Global_collection_names
        global Global_survey_title
        global Global_file_names

        Global_survey_title = request.POST.get('topic', False)

        if Global_test_flag == True:
            uid_str = 'test_4'
        else:
            uid_str = generate_uid()
        Global_survey_id = uid_str
        print('Uploaded survey id', Global_survey_id)

        for file_name in request.FILES:
            file = request.FILES[file_name]
            if not file.name:
                return JsonResponse({'error': 'No selected file'}, status=400)
            if file:
                # 规范化文件名，去除不合法字符，使用空格替换
                sanitized_filename = sanitize_filename_py(os.path.splitext(file.name)[0])
                file_extension = os.path.splitext(file.name)[1].lower()  # 统一扩展名为小写
                sanitized_filename = f"{sanitized_filename}{file_extension}"  # 加上扩展名

                # 构建文件存储路径
                # Replace spaces with underscores or another safe character if needed in paths
                # Here, spaces are kept as per requirement
                file_path = os.path.join('src', 'static', 'data', 'pdf', uid_str, sanitized_filename)
         
                
                # 检查文件是否存在
                if default_storage.exists(file_path):
                    # 删除原文件
                    default_storage.delete(file_path)
                
                # 保存新文件
                saved_file_name = default_storage.save(file_path, file)
                file_size = round(float(file.size) / 1024000, 2)

                # title = file.name
                # title_new = title.strip()
                # invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*','_']
                # for char in invalid_chars:
                #     title_new = title_new.replace(char, ' ')
                # print("============================")
                # print(title_new)
                # saved_file_name = f'./src/static/data/pdf/{uid_str}/{title_new}'


                collection_name, processed_file = process_file(saved_file_name, Global_survey_id)
                Global_collection_names.append(collection_name)
                Global_file_names.append(processed_file)
                filenames.append(processed_file)
                filesizes.append(file_size)
                print(filenames)
                print(filesizes)

        # if len(list(file_dict.keys())) > 0:
        #     file_name = list(file_dict.keys())[0]
        #     file_obj = file_dict[file_name]
        # else:
        #     is_valid_submission = False
        print("The global collection names are:", Global_collection_names)
        global Survey_dict
        survey_title = file_name.split('.')[-1].title()
        Survey_dict[uid_str] = survey_title

        new_file_name = Global_survey_id
        csvfile_name = new_file_name + '.'+ file_name.split('.')[-1]

        # Initialize an empty DataFrame
        json_data_pd = pd.DataFrame()

        # Define the path to the JSON files
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

        # Iterate over each JSON file
        for file_path in filtered_json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Extract necessary information
                title = data.get("title", "")
                abstract = data.get("abstract", "")
                authors = data.get("authors", "")
                introduction = data.get("introduction", "")

                # Append a new row with all information for a single paper
                # json_data_pd = json_data_pd.append({
                #     "reference paper title": title,
                #     "reference paper citation information (can be collected from Google scholar/DBLP)": authors,
                #     "reference paper abstract (Please copy the text AND paste here)": abstract,
                #     "reference paper introduction (Please copy the text AND paste here)": introduction,
                #     "reference paper doi link (optional)": "",
                #     "reference paper category label (optional)": ""
                # }, ignore_index=True)

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

                # Add title and abstract to the dictionary
                title_abstract_dict[title] = abstract

        # Save the DataFrame to a variable for further use
        input_pd = json_data_pd

        # Define the output path for the title-abstract JSON file
        output_path = f'./src/static/data/info/{Global_survey_id}/title_abstract_pairs.json'

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the title and abstract pairs to a JSON file
        with open(output_path, 'w') as outfile:
            json.dump(title_abstract_dict, outfile, indent=4, ensure_ascii=False)

        print(f'Title-abstract pairs have been saved to {output_path}')

        if ref_paper_num>0:
                
            ## change col name
            try:
                # required columns
                # input_pd["ref_title"] = input_pd["reference paper title"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid title')
                # input_pd['ref_title'] = ['_'.join(filename.split("_")[:-1]) for filename in filenames]
                input_pd['ref_title'] = [filename for filename in filenames]
                print(input_pd['ref_title'])
                print('++++++++++++++++++++++++++++++++++++++++++++++')
                input_pd["ref_context"] = [""]*ref_paper_num
                input_pd["ref_entry"] = input_pd["reference paper citation information (can be collected from Google scholar/DBLP)"]
                input_pd["abstract"] = input_pd["reference paper abstract (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid abstract')
                input_pd["intro"] = input_pd["reference paper introduction (Please copy the text AND paste here)"].apply(lambda x: clean_str(x) if len(str(x))>0 else 'Invalid introduction')

                # optional columns
                input_pd["ref_link"] = input_pd["reference paper doi link (optional)"].apply(lambda x: x if len(str(x))>0 else '')
                input_pd["label"] = input_pd["reference paper category label (optional)"].apply(lambda x: str(x) if len(str(x))>0 else '')
                #input_pd["label"] = input_pd["reference paper category id (optional)"].apply(lambda x: str(x) if len(str(x))>0 else '')
            except:
                print("Cannot convert the column name")
                is_valid_submission = False

            ## get cluster_num, check has_label_id
            stat_input_pd_labels = input_pd["label"].value_counts()
            #pdb.set_trace()
            clusters_topic_words = []
            if len(stat_input_pd_labels.keys())>1:
                cluster_num = len(stat_input_pd_labels.keys())
                clusters_topic_words = stat_input_pd_labels.keys().tolist()
                has_label_id = True
            else:
                #pdb.set_trace()
                cluster_num = 3
                
            global Survey_n_clusters
            Survey_n_clusters[uid_str] = cluster_num
            global Survey_Topic_dict
            Survey_Topic_dict[uid_str] = clusters_topic_words

            ## check has_ref_link
            if len(input_pd["ref_link"].value_counts().keys())>1:
                has_ref_link = True


            ## get keywords
            try:
                #pdb.set_trace()
                input_pd["topic_word"],input_pd["topic_bigram"],input_pd["topic_trigram"] = ['']*ref_paper_num,['']*ref_paper_num,['']*ref_paper_num
                # input_pd["topic_word"],input_pd["topic_bigram"],input_pd["topic_trigram"] = PosRank_get_top5_ngrams(input_pd)
                # input_pd["topic_word"],input_pd["topic_bigram"],input_pd["topic_trigram"] = abs_top5_unigram_list_list, abs_top5_bigram_list_list, abs_top5_trigram_list_list

                #Survey_Topic_dict[uid_str] = input_pd["topic_word"]
            except:
                print("Cannot select keywords")
                is_valid_submission = False
                #Survey_Topic_dict[uid_str] = []

            #pdb.set_trace()
            ## generate reference description
            try:
                # nltk.download('punkt')
                # nltk.download('averaged_perceptron_tagger')
                # nltk.download('wordnet')
                # nltk.download("maxent_treebank_pos_tagger")
                ref_desp_gen = ref_desp(input_pd)
                description_list = ref_desp_gen.ref_desp_generator()
                ref_desp_list=[]
                for ref_desp_set in description_list:
                    ref_desp_list.append(ref_desp_set[1])
                #pdb.set_trace()
                input_pd["description"]=ref_desp_list
            except:
                print("Cannot generate reference paper's description")
                is_valid_submission = False

            #pdb.set_trace()
            ## output tsv
            try:
                output_tsv_filename = "./src/static/data/tsv/" + new_file_name + '.tsv'

                #output_df = input_pd[["ref_title","ref_context","ref_entry","abstract","intro","description"]]
                output_df = input_pd[["ref_title","ref_context","ref_entry","abstract","intro","topic_word","topic_bigram","topic_trigram","description"]]
                print(output_df)

                if has_label_id == True:
                    output_df["label"]=input_pd["label"]
                else:
                    output_df["label"]=[""]*input_pd.shape[0]
                if has_ref_link == True:
                    output_df["ref_link"]=input_pd["ref_link"]
                else:
                    output_df["ref_link"]=[""]*input_pd.shape[0]

                #pdb.set_trace()
                output_df.to_csv(output_tsv_filename, sep='\t')
            except:
                print("Cannot output tsv")
                is_valid_submission = False
                #Survey_dict[Global_survey_id] = topic
                #Survey_Topic_dict[Global_survey_id] = [topic.lower()]
        else:
            # no record in submitted file
            is_valid_submission = False

        if is_valid_submission == True:
            if len(clusters_topic_words) == 0:
                references = output_df['ref_title'].tolist()
                ref_links = output_df['ref_link'].tolist()
                ref_ids = [i for i in range(output_df['ref_title'].shape[0])]

            elif len(clusters_topic_words)>0:
                references = []
                ref_links = []
                ref_ids = []
                for df in output_df.groupby('label'):
                    references.append(list(df[1]['ref_title']))
                    ref_links.append(list(df[1]['ref_link']))
                    ref_ids.append(df[1].index.tolist())
                    #pdb.set_trace()
                    #ref_ids.append(list(df[1]['ref_id']))

            ref_list = {'references':[i.title() for i in references],
                        'ref_links':ref_links,
                        'ref_ids':ref_ids,
                        'is_valid_submission':is_valid_submission,
                        "uid":uid_str,
                        "tsv_filename":output_tsv_filename,
                        'topic_words': clusters_topic_words,
                        'filenames': filenames,
                        'filesizes': filesizes,
                        'survey_id': Global_survey_id
                        }

        else:
            ref_list = {'references':[],'ref_links':[],'ref_ids':[],'is_valid_submission':is_valid_submission,"uid":uid_str,"tsv_filename":output_tsv_filename,'topic_words': [], 'filenames': filenames, 'filesizes': filesizes, 'survey_id': Global_survey_id}
            #ref_list = {'references':[],'ref_links':[],'ref_ids':[]}
        #pdb.set_trace()
        ref_list = json.dumps(ref_list)
        print(ref_list)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print("--- %s seconds used in processing files ---" % (time.time() - start_time))
        return HttpResponse(ref_list)

@csrf_exempt
def annotate_categories(request):
    # Global_survey_id = request.POST.get('uid', False)
    # global Global_survey_id
    # POST_dict = dict(request.POST)
    # topic_word_list = POST_dict['topic_word_list[]']
    # annotated_paper_ids_list_list= []
    # for i in range(len(POST_dict)-1):
    #     annotated_paper_ids_list_list.append([int(i) for i in POST_dict['ref_lists['+str(i)+'][]']])

    # assert(len(topic_word_list)==len(annotated_paper_ids_list_list))

    # tsvfile_name = Global_survey_id + '.tsv'
    # input_pd = pd.read_csv(TSV_PATH + tsvfile_name, sep = '\t')
    # #assert(len(annotated_topic_word_list)==input_pd.shape[0])

    # annotated_topic_word_list = ['']*input_pd.shape[0]
    # for category_id,annotated_paper_ids_list in enumerate(annotated_paper_ids_list_list):
    #     for annotated_paper_id in annotated_paper_ids_list:
    #         annotated_topic_word_list[annotated_paper_id] = topic_word_list[category_id]

    # input_pd['label'] = annotated_topic_word_list
    # #pdb.set_trace()
    # output_tsv_filename = TSV_PATH + tsvfile_name
    # os.remove(output_tsv_filename)
    # input_pd.to_csv(output_tsv_filename, sep='\t')

    html = generateOutlineHTML_qwen(Global_survey_id)
    print("The outline has been parsed successfully.")
    # print("The generated html: ", html)
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
def automatic_taxonomy(request):
    global Global_description_list, Global_df_selected, Global_cluster_names, Global_ref_list, Global_category_label, Global_collection_names_clustered
    ref_dict = dict(request.POST)
    print("The ref_dict is: ")
    print(ref_dict)
    ref_list = ref_dict['refs']
    query = ref_dict['taxonomy_standard'][0]
    query_list = generate_sentence_patterns(query)
    print(query_list)

    for name in Global_collection_names:
        # old
        # context = query_embeddings_new(name, query_list)

        # wza
        context, citation_data = query_embeddings_new_new(name, query_list)
        Global_citation_data.extend(citation_data)

        description = generate(context, query, name)
        Global_description_list.append(description)

    # Save citation data to file for debugging or reference
    citation_path = f'./src/static/data/info/{Global_survey_id}/citation_data.json'
    os.makedirs(os.path.dirname(citation_path), exist_ok=True)
    with open(citation_path, 'w') as outfile:
        json.dump(Global_citation_data, outfile, indent=4, ensure_ascii=False)

    # 定义文件名
    file_path = f'./src/static/data/tsv/{Global_survey_id}.tsv'

    # 读取现有文件并追加新列
    with open(file_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t')
        rows = list(reader)

    # 确保文件不为空
    if rows:
        # 获取表头并追加新列名
        headers = rows[0]
        headers.append('retrieval_result')

        # 更新数据行
        updated_rows = [headers]
        for row, description in zip(rows[1:], Global_description_list):
            row.append(description)
            updated_rows.append(row)

        # 写回原文件
        with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter='\t')
            writer.writerows(updated_rows)

        print('Updated file has been saved to', file_path)
    else:
        print('Input file is empty.')

    Global_ref_list = ref_list

    print('Categorization survey id', Global_survey_id)

    colors, category_label =  Clustering_refs(n_clusters=3) # fix with 3
    # colors, category_label, category_description = Clustering_refs_with_criteria(n_clusters=Survey_n_clusters[Global_survey_id], query=query)

    Global_category_label = category_label

    df_tmp = Global_df_selected.reset_index()
    df_tmp['index'] = df_tmp.index
    ref_titles = list(df_tmp.groupby(df_tmp['label'])['ref_title'].apply(list))
    ref_indexs = list(df_tmp.groupby(df_tmp['label'])['index'].apply(list))

    info = pd.read_json(f'./src/static/data/info/{Global_survey_id}/topic.json')
    category_label = info['KeyBERT'].to_list()
    category_label_summarized=[]

    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_id,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     token = os.getenv('HF_API_KEY'),
    #     device_map="auto",
    # )
    for i in range(len(category_label)):
        messages = [
            {"role": "system", "content": "You are a research topic summarizer and your task is to summarize the following keywords into one phrase as a cluster name within five words research topic. Noted that you are only allowed to output one research topic in total.\
                For example, the keywords are: Neural Networks_Deep Learning_Convolutional Layers_Backpropagation_Gradient Descent_Activation Functions_Supervised Learning_Model Training_Data Preprocessing_Overfitting\
                and your response should be: 'Deep Learning Techniques' these three words purely."},
            {"role": "user", "content": "The keywords are: " + str(category_label[i])},
        ]
        outputs = Global_pipeline(
            messages,
            max_new_tokens=256,
        )
        print(outputs[0]["generated_text"][-1]['content'])
        if len(outputs[0]["generated_text"][-1]['content'].split())>8:
            category_label_summarized.append(category_label[i][0])
        else:
            category_label_summarized.append(outputs[0]["generated_text"][-1]['content'].replace("'",'').replace('"','').strip())
    tsv_path = f'./src/static/data/tsv/{Global_survey_id}.tsv'

    # comment this for old version

    #V1
    # category_label_summarized = generate_cluster_name_qwen_sep(tsv_path, Global_survey_title)
    # category_label_summarized = refine_cluster_name(category_label_summarized, Global_survey_title)

    #V2
    category_label_summarized = generate_cluster_name_new(tsv_path, Global_survey_title)   
    Global_cluster_names = category_label_summarized

    print(category_label)
    print('+++++++++++++++++++++++++++++')
    print(category_label_summarized)


    cate_list = {
        'colors': colors,
        'category_label': category_label_summarized,
        'survey_id': Global_survey_id,
        'ref_titles': [[i.title() for i in j] for j in ref_titles],
        'ref_indexs': ref_indexs
    }
    print(cate_list)
    cate_list = json.dumps(cate_list)

    cluster_info = {category_label_summarized[i]:ref_titles[i] for i in range(len(category_label_summarized))}
    for key, value in cluster_info.items():
        temp = [legal_pdf(i) for i in value]
        cluster_info[key] = temp
        Global_collection_names_clustered.append(temp)
    cluster_info_path = f'./src/static/data/info/{Global_survey_id}/cluster_info.json'
    with open(cluster_info_path, 'w') as outfile:
        json.dump(cluster_info, outfile, indent=4, ensure_ascii=False)

    
    outline_generator = OutlineGenerator(Global_pipeline, Global_df_selected, Global_cluster_names)
    outline_generator.get_cluster_info()
    messages, outline = outline_generator.generate_outline_qwen(Global_survey_title)
    print(outline)

    outline_json = {'messages':messages, 'outline': outline}
    output_path = TXT_PATH + Global_survey_id + '/outline.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as outfile:
        json.dump(outline_json, outfile, indent=4, ensure_ascii=False)

    return HttpResponse(cate_list)

@csrf_exempt
def save_updated_cluster_info(request):
    if request.method == 'POST':  # 确保只处理 POST 请求
        try:
            # 获取请求数据
            data = json.loads(request.body)  # 从 request.body 中解析 JSON 数据
            survey_id = Global_survey_id
            updated_cate_list = data.get('updated_cate_list')

            if not survey_id or not updated_cate_list:
                return JsonResponse({"error": "Missing survey_id or updated_cate_list"}, status=400)

            # 构造保存路径
            save_dir = os.path.join('./src/static/data/info/', str(survey_id))
            os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
            save_path = os.path.join(save_dir, 'cluster_info_updated.json')

            # 保存到 JSON 文件
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

@csrf_exempt  # 如果前端未发送 CSRF token，可以暂时使用它，但建议在生产环境中避免
def save_outline(request):
    if request.method == 'POST':
        try:
            # 从 request.body 获取 JSON 数据
            data = json.loads(request.body)
            updated_outline = data.get('outline', [])  # 从 JSON 数据中获取 "outline" 字段

            # 构造要保存的 JSON 数据
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

            # 动态生成文件路径
            file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'txt', Global_survey_id,'outline.json')

            # 确保文件目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(outline_data, file, indent=4, ensure_ascii=False)
            
            html = generateOutlineHTML_qwen(Global_survey_id)

            return JsonResponse({"status": "success", "html": html})
        except Exception as e:
            # 捕获错误并返回
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    else:
        return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)

@csrf_exempt
def select_sections(request):

    sections = request.POST
    # print(sections)

    survey = {}

    for k,v in sections.items():
        print(Survey_dict[Global_survey_id])
        # if k == "title":
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
        # colors, category_label, category_description = Clustering_refs(n_clusters=Survey_n_clusters[Global_survey_id])
        # for ref in Global_df_selected['ref_entry']:
        #     entry = str(ref).encode('utf-8')
        #     survey['references'].append(entry)

    survey_dict = json.dumps(survey)

    return HttpResponse(survey_dict)

@csrf_exempt
def get_survey(request):
    survey_dict = get_survey_text()
    survey_dict = json.dumps(survey_dict)
    return HttpResponse(survey_dict)
    
# @csrf_exempt
# def get_survey_id(request):
#     # global Global_survey_id, Global_survey_title, Global_collection_names, Global_pipeline
#     global Global_survey_id, Global_survey_title, Global_collection_names, Global_pipeline, Global_citation_data

#     # old
#     # generateSurvey_qwen(Global_survey_id, Global_survey_title, Global_collection_names_clustered, Global_pipeline)
    
#     # wza
#     generateSurvey_qwen_new(Global_survey_id, Global_survey_title, Global_collection_names_clustered, Global_pipeline, Global_citation_data)

#     return JsonResponse({"survey_id": Global_survey_id})

@csrf_exempt
def get_survey_id(request):
    global Global_survey_id, Global_survey_title, Global_collection_names_clustered, Global_pipeline, Global_citation_data

    # 调用生成Survey的函数（需事先确保generateSurvey_qwen_new的定义）
    # generateSurvey_qwen_new会生成generated_result.json文件，其中包括"content"字段的完整survey文本
    generateSurvey_qwen_new(Global_survey_id, Global_survey_title, Global_collection_names_clustered, Global_pipeline, Global_citation_data)

    return JsonResponse({"survey_id": Global_survey_id})

@csrf_exempt
def generate_pdf(request):
    if request.method == 'POST':
        survey_id = request.POST.get('survey_id', '')
        # 获取前端传递的 HTML 内容
        markdown_content = request.POST.get('content', '')
<<<<<<< HEAD
=======
        # print("The global collection names:")
        # print(Global_collection_names)
        # print("The global file names:")
        # print(Global_file_names)
        # print("="*24)
        markdown_content = finalize_survey_paper(markdown_content,Global_collection_names, Global_file_names)
        survey_id = request.POST.get('survey_id', '')
>>>>>>> 9c99fed7376389523a9cf2534cda8d35589713e3

        # 设置 Markdown 文件的保存路径
        markdown_dir = f'./src/static/data/info/{survey_id}/'
        markdown_filename = f'survey_{survey_id}_vanilla.md'
        markdown_filepath = os.path.join(markdown_dir, markdown_filename)

        # 确保目标目录存在
        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir)
            print(f"Directory '{markdown_dir}' created.")
        else:
            print(f"Directory '{markdown_dir}' already exists.")

        # 保存 Markdown 内容到文件
        with open(markdown_filepath, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(markdown_content)
        print(f"Markdown content saved to: {markdown_filepath}")

        markdown_content = finalize_survey_paper(markdown_content, Global_collection_names, Global_file_names)

        # 设置 Markdown 文件的保存路径1
        markdown_dir = f'./src/static/data/info/{survey_id}/'
        markdown_filename = f'survey_{survey_id}_processed.md'
        markdown_filepath = os.path.join(markdown_dir, markdown_filename)

        # 确保目标目录存在1
        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir)
            print(f"Directory '{markdown_dir}' created.")
        else:
            print(f"Directory '{markdown_dir}' already exists.")

        # 保存 Markdown 内容到文件1
        with open(markdown_filepath, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(markdown_content)
        print(f"Markdown content saved to: {markdown_filepath}")

        # 配置 PDF 文件的保存路径
        pdf_filename = f'survey_{survey_id}.pdf'
        pdf_dir = './src/static/data/results'
        pdf_filepath = os.path.join(pdf_dir, pdf_filename)

        # 检查并创建 results 目录
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            print(f"Directory '{pdf_dir}' created.")
        else:
            print(f"Directory '{pdf_dir}' already exists.")

        # 打印文件保存路径信息
        print(f"PDF will be saved to: {pdf_filepath}")

        # 使用 markdown_pdf 库生成 PDF
        pdf = MarkdownPdf()
        pdf.meta["title"] = "Survey Results"  # 设置 PDF 的元数据
        pdf.add_section(Section(markdown_content, toc=False))  # 添加 Markdown 内容，不生成目录
        pdf.save(pdf_filepath)  # 将 PDF 保存到文件

        # 打开文件并将其作为响应返回
        with open(pdf_filepath, 'rb') as pdf_file:
            response = HttpResponse(pdf_file.read(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{pdf_filename}"'
            return response

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
    print('REFERENCES FOR GENERATING SURVEY CONTENT', refs)
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
    df = pd.read_csv(TSV_PATH + Global_survey_id + '.tsv', sep='\t', index_col=0, encoding='utf-8')
    print(df.describe())
    print(df)
    print(Global_ref_list)
    df_selected = df.iloc[Global_ref_list]
    
    print(df_selected)

    ## update cluster labels and keywords
    df_selected, colors = clustering(df_selected, n_clusters, Global_survey_id)

    global Global_df_selected
    Global_df_selected = df_selected
    category_description = [0]*len(colors)
    category_label = [0]*len(colors)

    return colors, category_label
    # return 1,0,1

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

# wza
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

# paper_with_non_standard_citations = """
# This is a paper text with non-standard citations like [collection name1] and [collection name2].
# Another reference to [collection name1] is here. And then we see [collection name3].
# """

# normalized_paper, citation_mapping = normalize_citations_with_mapping(paper_with_non_standard_citations)
# print("Normalized Paper:")
# print(normalized_paper)
# print("\nCitation Mapping:")
# print(citation_mapping)


# wza
def generate_references_section(citation_mapping, collection_pdf_mapping):
    print("The citation mapping is:")
    print(citation_mapping)
    print("The collection pdf mapping is:")
    print(collection_pdf_mapping)
    print("-" * 24)
    
    references = ["# References"]  # 生成引用部分
    for num in sorted(citation_mapping.keys()):
        collection_name = citation_mapping[num]
        pdf_name = collection_pdf_mapping.get(collection_name, "Unknown PDF")
        if pdf_name.endswith(".pdf"):
            pdf_name = pdf_name[:-4]
        # 在每一行末尾添加两个空格以确保换行
        references.append(f"[{num}] {pdf_name}  ")

    return "\n".join(references)

#wzawza
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

#wzawza
def finalize_survey_paper(paper_text, 
                          Global_collection_names, 
                          Global_file_names):

    print("=== finalize_survey_paper is called ===")
    print("Global_collection_names =", Global_collection_names)
    print("Global_file_names =", Global_file_names)
    print("--- original paper_text ---")
    print(paper_text)

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


# 调用
# paper_text = """
# In this paper, we discuss various approaches [collection_name_1]. 
# Some of them are advanced in [collection_name_2].
# """

# Global_collection_names = ["collection_name_1", "collection_name_2"]
# Global_file_names = ["paper1.pdf", "paper2.pdf"]

# result = finalize_survey_paper(paper_text, Global_collection_names, Global_file_names)
# print(result)