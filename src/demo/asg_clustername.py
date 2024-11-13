import os
import pandas as pd
import openai
import re  # Import the regular expressions module
from openai import OpenAI

def generate_cluster_name_qwen_sep(tsv_path, survey_title):
    global Global_survey_id
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
            model="Qwen2.5-72B-Instruct",
            max_tokens=768,
            temperature=0.5,
            stop="<|im_end|>",
            stream=True,
            messages=messages
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
    print("The generated cluster names are:")
    print(result)
    return result  # This will be a list with three elements

# Example usage:
# result = generate_cluster_name_qwen_sep('path_to_your_file.tsv', 'Your Survey Title')
# print(result)  # Output might look like ["Cluster One", "Cluster Two", "Cluster Three"]