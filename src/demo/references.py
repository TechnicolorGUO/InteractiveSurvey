import os
import re
from openai import OpenAI
# from .asg_retriever import Retriever


def getQwenClient(): 
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
        max_tokens=1536,
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

def generate_references(papers_info, client):

    # In-Context Learning
    examples = '''
Example1:
Authors: Armen Aghajanyan, Armen Aghajanyan, Anchit Gupta, Akshat Shrivastava, Xilun Chen, Luke Zettlemoyer, and Sonal Gupta
Title: Muppet: Massive multi-task representations with pre-finetuning
Reference: Armen Aghajanyan, Anchit Gupta, Akshat Shrivastava, Xilun Chen, Luke Zettlemoyer, and Sonal Gupta. Muppet: Massive multi-task representations with pre-finetuning

Example2:
Authors: Ari Holtzman1, Peter West222, Vered Shwartz3, Yejin Choi4, Luke Zettlemoyer12001
Title:  Surface form competition: Why the highest probability answer isn't always right.
Reference: Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, Luke Zettlemoyer. Surface form competition: Why the highest probability answer isn't always right.

Example3:
Authors: Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, Giri Anantharaman, Xian Li, Shuohui Chen, Halil Akin, Mandeep Baines, Louis Martin, Xing Zhou, Punit Singh Koura, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Mona Diab, Zornitsa Kozareva, Ves Stoyanov
Title: Efficient large scale language modeling with mixtures of experts.
Reference: Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, et al. Efficient large scale language modeling with mixtures of experts.
'''

    prompt = f'''
Based on the following examples, generate the references based on the provided paper information.
The generated references should be clear, legal and properly formatted.
If the authors are many, list the first few authors followed by "et al.".

Please include the "Reference:" label before each reference as shown in the examples.

{examples}
Now, please generate the references:

'''

    for idx, paper in enumerate(papers_info):
        authors = paper['authors']
        title = paper['title']
        prompt += f'''
Paper{idx+1}:
Authors: {authors}
Title: {title}
Reference:'''
    
    response = generateResponse(client, prompt)
    # print("Generated Response:", response)

    # Extract references from response
    references = []
    pattern = r'Reference:(.*?)(?=\n\n|$)'
    matches = re.findall(pattern, response, re.S)
    
    for match in matches:
        reference = match.strip()
        if reference:
            references.append(reference)

    return references

# Example usage
if __name__ == '__main__':
    client = getQwenClient()
    papers_info = [
        {'authors': 'Jie Ren 1 , Han Xu 1 , Yiding Liu 2 , Yingqian Cui 1 , Shuaiqiang Wang 2 Dawei Yin 2 , Jiliang Tang 1 1 Michigan State University, Baidu Inc. {renjie3, xuhan1, cuiyingq, tangjili}@msu.edu liuyiding.tanh@gmail.com, shqiang.wang@gmail.com, yindawei@acm.org', 'title': 'A Robust Semantics-based Watermark for Large Language Models against Paraphrasing'},
        {'authors': 'Alice Cooper, Bob Dylan', 'title': 'Exploring Quantum Computing'},
        {'authors': 'Charlie Evans, Diana Foster, Ethan Green, Fiona Harris, George King', 'title': 'An Introduction to Bioinformatics'}
    ]

    # papers_info = [
    #     {'authors': 'Alice Cooper, Bob Dylan', 'title': 'Exploring Quantum Computing'},
    #     {'authors': 'Charlie Evans, Diana Foster, Ethan Green, Fiona Harris, George King', 'title': 'An Introduction to Bioinformatics'},
    #     {'authors': 'Sewon Min, Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, Giri Anantharaman, Xian Li, Shuohui Chen, Halil Akin, Mandeep Baines, Louis Martin, Xing Zhou, Punit Singh Koura, Brian OHoro, Jeff Wang, Luke Zettlemoyer, Mona Diab, Zornitsa Kozareva, Ves Stoyanov', 'title': 'Efficient large scale language modeling with mixtures of experts'}
    # ]
    # references = generate_references(papers_info, client)
    # print(references)
    # print("\n\n\n\n\n")

    # print("Generated References:\n")
    # for ref in references:
    #     print(ref)
