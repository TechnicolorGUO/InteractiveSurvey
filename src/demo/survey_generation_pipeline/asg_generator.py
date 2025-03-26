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
        # defaults to os.environ.get("OPENAI_API_KEY")
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
    # Stream the response to console
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
        # defaults to os.environ.get("OPENAI_API_KEY")
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
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    response = generateResponse(client, template)
    return response

# no need for api
def extract_query_list(text):
    pattern = re.compile(
        r'\[\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"\s*\]'
    )
    match = pattern.search(text)
    if match:
        return match.group(0)
    return None

# # test
# if __name__ == "__main__":
#     client = getQwenClient()
#     keyword = "methods"
#     query_list = generate_sentence_patterns(keyword, client)
#     print("Generated Query List:", query_list)

#     context = """
#     on the learning process. Specifically, it first divides the text embedding into real and imaginary parts in a complex space. Then, it follows the division rule in complex space to compute the angle difference between two text embeddings. After normalization, the angle difference becomes an//
#     and syntactic information in language, which broadly affects the performance of downstream tasks, such as text classification (Li et al., 2021), sentiment analysis (Suresh & Ong, 2021; Zhang et al., 2022), semantic matching (Grill et al., 2020; Lu et al., 2020), clustering (Reimers & Gurevych,//
#     becomes an objective to be optimized. It is intuitive to optimize the normalized angle difference, because if the normalized angle difference between two text embeddings is smaller, it means that the two text embeddings are closer to each other in the complex space, i.e., their similarity is//
#     We first experimented with both short and longtext datasets and showed that AnglE outperforms the SOTA STS models in both transfer and nontransfer STS tasks. For example, AnglE shows an average Spearman correlation of 73.55% in nontransfer STS tasks, compared to 68.03% for SBERT. Then, an ablation//
#     Then, an ablation study shows that all components contribute positively to AnglE’s superior performance. Next, we discuss the domainspecific scenarios with limited annotated data that are challenging for AnglElike supervised STS, where it is observed that AnglE can work well with LLMsupervised//
#     samples that are dissimilar are selected from different texts within the same minibatch (inbatch negatives). However, supervised negatives are underutilized, and the correctness of inbatch negatives is difficult to guarantee without annotation, which can lead to performance degradation. Although//
#     For supervised STS (Reimers & Gurevych, 2019; Su, 2022), most efforts to date employed the cosine function in their training objective to measure the pairwise semantic similarity. However, the cosine function has saturation zones, as shown in Figure 1. It can impede the optimization due to the//
#     """
#     result = generate(context, client, keyword)
#     print("Generated Result:", result)

'''
Generated Query List: [
    "The methods employed in this study include...",
    "To achieve the research objectives, the following methods were utilized...",
    "This paper employs a combination of methods to...",
    "The primary methods used in this research are...",
    "In this study, we adopted the following methods...",
    "The methods section outlines the procedures and techniques used...",
    "The research methods are...",
    "In this paper, we...",
    "We propose...",
    "Our method is..."
]
Generated Result: The paper employs a method that divides text embeddings into real and imaginary parts within a complex space, computes the angle difference between these embeddings according to complex space division rules, and optimizes the normalized angle difference to enhance semantic similarity. This approach is tested on both short and long text datasets, showing superior performance in semantic textual similarity (STS) tasks compared to state-of-the-art models. An ablation study confirms the positive contribution of each component of the method.
'''