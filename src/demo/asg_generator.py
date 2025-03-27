from openai import OpenAI
import transformers
import os
import re
import ast
import json
import base64

def getQwenClient(): 
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_base = os.environ.get("OPENAI_API_BASE")

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def generateResponse(client, prompt):
    chat_response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        max_tokens=768,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=[{"role": "user", "content": prompt}]
    )

    text = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    return text

def generate_sentence_patterns(keyword, num_patterns=5, temp=0.7):
    template = f"""
You are a helpful assistant that provides only the output requested, without any additional text.

Please generate {num_patterns} commonly used sentence templates in academic papers to describe the '{keyword}'.
- Do not include any explanations, sign-offs, or additional text.
- The list should be in the following format:
[
    "First template should be here",
    "Second template should be here",
    ...
]

Begin your response immediately with the list, and do not include any other text.
"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    response = generateResponse(client, template)
    return response

def generate(context, keyword, paper_title, temp=0.7):
    template = f"""
Context:
{context}
------------------------------------------------------------
Based on the above context, answer the question: What {keyword} are mentioned in the paper {paper_title}?
Please provide a direct answer in one paragraph, no longer than 100 words. 

If the context provides enough information, answer strictly based on it. 
If the context provided does not contain any specified {keyword}, deduce and integrate your own opinion as if the {keyword} were described in the context. 
Ensure that your answer remains consistent with the style and format of the provided context, as if the information you provide is naturally part of it.
------------------------------------------------------------
Answer:
The {keyword} mentioned in this paper discuss [Your oberservation or opinion]...
"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    response = generateResponse(client, template)
    return response

def extract_query_list(text):
    pattern = re.compile(
        r'\[\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*\]'
    )
    match = pattern.search(text)
    if match:
        return match.group(0)
    return None