import os
import pandas as pd
import re  # Import the regular expressions module
from openai import OpenAI
import ast

def generate_cluster_name_qwen_sep(tsv_path, survey_title):
    data = pd.read_csv(tsv_path, sep='\t')
    
    # Define the system prompt once, outside the loop
    system_prompt = f'''You are a research assistant working on a survey paper. The survey paper is about "{survey_title}". \
    '''
    
    result = []  # Initialize the result list

    for i in range(3):  # Assuming labels are 0, 1, 2
        sentence_list = []  # Reset sentence_list for each label
        for j in range(len(data)):
            if data['label'][j] == i:
                sentence_list.append(data['retrieval_result'][j])
        
        # Convert the sentence list to a string representation
        user_prompt = f'''
        Given a list of descriptions of sentences about an aspect of the survey, you need to use one phrase (within 8 words) to summarize it and treat it as a section title of your survey paper. \
Your response should be a list with only one element and without any other information, for example, ["Post-training of LLMs"]  \
Your response must contain one keyword of the survey title, unspecified or irrelevant results are not allowed. \
The description list is:{sentence_list}'''
        
        messages = [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt},
        ]
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")
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
        
        # Stream the response to a single text string
        text = ""
        for chunk in chat_response:
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        
        # Use regex to extract the first content within []
        match = re.search(r'\[(.*?)\]', text)
        if match:
            cluster_name = match.group(1).strip()  # Extract and clean the cluster name
            # 去除集群名称两侧的引号（如果存在）
            cluster_name = cluster_name.strip('"').strip("'")
            result.append(cluster_name)
        else:
            result.append("No Cluster Name Found")  # Handle cases where pattern isn't found
    # print("The generated cluster names are:")
    # print(result)
    return result  # This will be a list with three elements
    
# Example usage:
# result = generate_cluster_name_qwen_sep('path_to_your_file.tsv', 'Your Survey Title')
# print(result)  # Output might look like ["Cluster One", "Cluster Two", "Cluster Three"]

def refine_cluster_name(cluster_names, survey_title):
    cluster_names = str(cluster_names)  # Convert to string to handle list input
    # Define the system prompt to set the context
    system_prompt = f'''You are a research assistant tasked with optimizing and refining a set of section titles for a survey paper. The survey paper is about "{survey_title}". 
'''
    
    # Construct the user prompt, including all cluster names
    user_prompt = f'''
Here is a set of section titles generated for the survey topic "{survey_title}":
{cluster_names}
Please ensure that all cluster names are coherent and consistent with each other, and that each name is clear, concise, and accurately reflects the corresponding section.
Notice to remove the overlapping information between the cluster names.
Each cluster name should be within 8 words and include a keyword from the survey title.
Response with a list of section titles in the following format without any other irrelevant information,
For example, ["Refined Title 1", "Refined Title 2", "Refined Title 3"]
'''
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    try:
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
    
        # Stream the response and concatenate into a complete text
        text = ""
        for chunk in chat_response:
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content

        # print("The raw response text is:")
        # print(text)
    
        # Use regex to extract content within square brackets
        match = re.search(r'\[(.*?)\]', text)
        if match:
            refined_cluster_names = match.group(1).strip()  # Extract and clean the cluster name
        else:
            refined_cluster_names = [
                survey_title + ": Definition",
                survey_title + ": Methods",
                survey_title + ": Evaluation"
            ]  # Handle cases where pattern isn't found
    
    except Exception as e:
        print(f"An error occurred while refining cluster names: {e}")
        refined_cluster_names = ["Refinement Error"] * len(cluster_names)
    
    refined_cluster_names = ast.literal_eval(refined_cluster_names)  # Convert string to list
    
    # print("The refined cluster names are:")
    # print(refined_cluster_names)
    return refined_cluster_names  # Returns a list with the refined cluster names、




def generate_cluster_name_new(tsv_path, survey_title, cluster_num = 3):
    data = pd.read_csv(tsv_path, sep='\t')
    desp=[]


    for i in range(cluster_num):  # Assuming labels are 0, 1, 2
        sentence_list = []  # Initialize the sentence list
        for j in range(len(data)):
            if data['label'][j] == i:
                sentence_list.append(data['retrieval_result'][j])
        desp.append(sentence_list)

    system_prompt = f'''
    You are a research assistant working on a survey paper. The survey paper is about "{survey_title}". '''
    
    cluster_info = "\n".join([f'Cluster {i+1}: "{desp[i]}"' for i in range(cluster_num)])

    user_prompt = f'''
    Your task is to generate {cluster_num} distinctive cluster names (e.g., "Pre-training of LLMs") of the given clusters of reference papers, each reference paper is described by a sentence.

    The clusters of reference papers are: 
    {cluster_info}

    Your output should be a single list of {cluster_num} cluster names, e.g., ["Pre-training of LLMs", "Fine-tuning of LLMs", "Evaluation of LLMs"]
    Do not output any other text or information.
    '''

    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt},
    ]
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
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
    
    # Stream the response to a single text string
    text = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    # print("The raw response text is:")
    # print(text)

    # Use regex to extract content within square brackets
    match = re.search(r'\[(.*?)\]', text)
    if match:
        refined_cluster_names = match.group(1).strip()  # Extract and clean the cluster name
    else:
        predefined_sections = [
            "Definition", "Methods", "Evaluation", "Applications",
            "Challenges", "Future Directions", "Comparisons", "Case Studies"
        ]
        
        # 根据 cluster_num 选择前 cluster_num 个预定义类别
        refined_cluster_names = [
            f"{survey_title}: {predefined_sections[i]}" for i in range(cluster_num)
        ]
    
    refined_cluster_names = ast.literal_eval(refined_cluster_names)  # Convert string to list
    
    # print("The refined cluster names are:")
    # print(refined_cluster_names)
    return refined_cluster_names  # Returns a list with the refined cluster names、


if __name__ == "__main__":
    refined_result = refine_cluster_name(["Pre-training of LLMs", "Fine-tuning of LLMs", "Evaluation of LLMs"], 'Survey of LLMs')
    # print(refined_result)
        
    