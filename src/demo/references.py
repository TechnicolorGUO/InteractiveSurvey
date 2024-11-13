import os
import re
from openai import OpenAI
# from .asg_retriever import Retriever


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
The generated references should be clear, legal.
If the authors are so many, then you can just list the first few authors and add "et al." at the end.
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

    # Extract references from response
    references = []
    pattern = r'Paper\s*\d+:.*?Reference:(.*?)(?=\nPaper\s*\d+:|$)'
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
        {'authors': 'Alice Cooper2123323, Bob Dylan88888', 'title': 'Exploring Quantum Computing12201'},
        {'authors': '999Charlie Evans, Diana Foster, Ethan Green, Fiona Harris, George King', 'title': 'An Introduction to Bioinformatics'},
        {'authors': 'Sewon Min, Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, Giri Anantharaman, Xian Li, Shuohui Chen, Halil Akin, Mandeep Baines, Louis Martin, Xing Zhou, Punit Singh Koura, Brian OHoro, Jeff Wang, Luke Zettlemoyer, Mona Diab, Zornitsa Kozareva, Ves Stoyanov', 'title': 'Efficient large scale language modeling with mixtures of experts'}
    ]
    references = generate_references(papers_info, client)
    print(references)
    # for ref in references:
    #     print(ref)
