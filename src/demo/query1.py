import os
from openai import OpenAI
from datetime import datetime, timedelta
import re

def generate_abstract_qwen(topic):
    
    # Initialize the OpenAI client using environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    
    ###########################
    # Step 1: Generate a survey abstract for the given topic.
    ###########################
    system_prompt_abstract = """
You are a skilled research survey writer. Your task is to generate a survey abstract on the given topic. The abstract should cover the main challenges, key concepts, and research directions associated with the topic. Write in clear, concise academic English.
"""
    user_prompt_abstract = f"""
Topic: {topic}

Please generate a comprehensive survey abstract for this topic. Include discussion of core challenges, key terminologies, and emerging methodologies that are critical in the field. The total length of the abstract should be around 300–500 words.
"""
    messages_abstract = [
        {"role": "system", "content": system_prompt_abstract},
        {"role": "user", "content": user_prompt_abstract}
    ]
    
    abstract_response = client.chat.completions.create(
        model="Qwen2.5-72B-Instruct",
        max_tokens=2048,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=messages_abstract
    )
    
    abstract_text = ""
    for chunk in abstract_response:
        if chunk.choices[0].delta.content:
            abstract_text += chunk.choices[0].delta.content
    abstract_text = abstract_text.strip()
    print("The abstract is:", abstract_text)

    return abstract_text

def generate_entity_lists_qwen(topic, abstract_text):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    system_prompt_abstract = f"""
    You are an AI assistant specializing in natural language processing and entity recognition. Your task is to extract key entities and core concepts from a given abstract based on a specified topic.  

    You should return two distinct lists:  
    1. **Entity list**: Entities that are synonymous or closely related to the given topic. These should be concise (no more than two words) and simplified to their root forms (e.g., removing suffixes like "-ing", "-ed").  
    2. **Concept list**: Core concepts from the abstract that are highly relevant to the topic. These should also be concise (no more than two words) and in their simplest form.  

    Ensure that your response follows this exact format:
    Entity list: [entity1, entity2, entity3, ...]
    Concept list: [concept1, concept2, concept3, ...]
    Do not include any explanations or additional text.  

    ### **Example**  
    #### **Input:**  
    Topic: Large Language Models  
    Abstract: Ever since the Turing Test was proposed in the 1950s, humans have explored the mastering of language intelligence by machine. Language is essentially a complex, intricate system of human expressions governed by grammatical rules. It poses a significant challenge to develop capable artificial intelligence (AI) algorithms for comprehending and grasping a language. As a major approach, language modeling has been widely studied for language understanding and generation in the past two decades, evolving from statistical language models to neural language models. Recently, pre-trained language models (PLMs) have been proposed by pretraining Transformer models over large-scale corpora, showing strong capabilities in solving various natural language processing (NLP) tasks. Since the researchers have found that model scaling can lead to an improved model capacity, they further investigate the scaling effect by increasing the parameter scale to an even larger size. Interestingly, when the parameter scale exceeds a certain level, these enlarged language models not only achieve a significant performance improvement, but also exhibit some special abilities (e.g., in-context learning) that are not present in small-scale language models (e.g., BERT). To discriminate the language models in different parameter scales, the research community has coined the term large language models (LLM) for the PLMs of significant size (e.g., containing tens or hundreds of billions of parameters). Recently, the research on LLMs has been largely advanced by both academia and industry, and a remarkable progress is the launch of ChatGPT (a powerful AI chatbot developed based on LLMs), which has attracted widespread attention from society. The technical evolution of LLMs has been making an important impact on the entire AI community, which would revolutionize the way how we develop and use AI algorithms. Considering this rapid technical progress, in this survey, we review the recent advances of LLMs by introducing the background, key findings, and mainstream techniques. In particular, we focus on four major aspects of LLMs, namely pre-training, adaptation tuning, utilization, and capacity evaluation. Furthermore, we also summarize the available resources for developing LLMs and discuss the remaining issues for future directions. This survey provides an up-to-date review of the literature on LLMs, which can be a useful resource for both researchers and engineers.  

    #### **Expected Output:**
    "entity list": ["language model", "plm", "large language", "llm"]  
    "concept list": ["turing", "language intelligence", "ai", "generation", "statistical", "neural", "pre-train", "transformer", "corpora", "nlp", "in-context", "bert", "chatgpt", "adaptation", "utilization"]
    Make sure to strictly follow this format in your response.
    """

    user_prompt_abstract = f"""
    Topic: {topic}  
    Abstract: {abstract_text}  

    Based on the given topic and abstract, extract the following:  
    1. A **list of entities** that are synonymous or closely related to the topic. Keep each entity under two words and in its simplest form.  
    2. A **list of core concepts** from the abstract that are highly relevant to the topic. Keep each concept under two words and in its simplest form.     
    """

    messages_abstract = [
        {"role": "system", "content": system_prompt_abstract},
        {"role": "user", "content": user_prompt_abstract}
    ]
    
    entity_response = client.chat.completions.create(
        model="Qwen2.5-72B-Instruct",
        max_tokens=2048,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=messages_abstract
    )
    
    entity_list = ""
    for chunk in entity_response:
        if chunk.choices[0].delta.content:
            entity_list += chunk.choices[0].delta.content
    entity_list = entity_list.strip()
    print("The entity lists are:", entity_list)

    return entity_list


def generate_query_qwen(topic):
    # Calculate date range for the arXiv query (last 5 years)
    abstract_text = generate_abstract_qwen(topic)
    entity_list = generate_entity_lists_qwen(topic, abstract_text)
    today = datetime.now()
    five_years_ago = today - timedelta(days=10 * 365)  # approximate calculation
    start_date = five_years_ago.strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')


    # System prompt: Focus on how to extract keywords from the abstract.
    system_prompt_query = """
    You are a research assistant specializing in constructing effective arXiv search queries. Your task is to generate a structured search query using **pre-extracted entity and concept lists** from a given abstract. Follow these instructions exactly:

    1. **Input Data:**
    - **Entity List:** A list of entities that are synonymous or closely related to the given topic.
    - **Concept List:** A list of core concepts from the abstract that are highly relevant to the topic.

    2. **Ensure Minimum Keyword Count:**
    - **Entity List** must contain at least **5** terms. If there are fewer, intelligently supplement additional relevant terms.
    - **Concept List** must contain **12-15** terms. If there are fewer, intelligently supplement additional relevant terms.

    3. **Standardize Formatting:**
    - Convert all terms to their **base form** and ensure they end with a wildcard `*`.  
        - Examples: `verification → verif*`, `optimization → optim*`, `retrieval → retriev*`, `embedding → embed*`
    - All terms must be **in lowercase**.

    4. **Construct the Final Query:**
    - The query must follow this exact structure:
        ```
        (abs:"<Entity1*>" OR abs:"<Entity2*>" OR abs:"<Entity3*>" OR abs:"<Entity4*>" OR abs:"<Entity5*>") AND 
        (abs:"<Concept1*>" OR abs:"<Concept2*>" OR ... OR abs:"<Concept12*>")
        ```
    - **Entities are grouped together using `OR` in the first part.**
    - **Concepts are grouped together using `OR` in the second part.**
    - **The two groups are combined using `AND`.**
    - **Do not include any explanations or extra text. Output only the final query.**
    """

    # User prompt: Provide examples of topics with corresponding query formats.
    # User prompt: Provide examples of topics with corresponding query formats.
# User prompt: Uses pre-extracted entities and concepts, ensures minimum count, and applies stemming + wildcards.
    user_prompt_query = f"""
    Below are the pre-extracted keywords for constructing the final arXiv query.

    **Topic:** {topic}  
    **Entity list and Concept list:** {entity_list}

    ### **Processing Rules Applied:**
    - **Ensure at least 5 entities** (if fewer, supplement additional relevant terms).
    - **Ensure 12-15 concepts** (if fewer, supplement additional relevant terms).
    - **Convert all terms to base form and append wildcard `*`.**
    - **Output only the final query with no extra text.**

    ### **Example Query Format:**

    1. **Topic:** Large Language Models  
    **Transformed Entity List:** ["languag model*", "plm*", "larg languag*", "llm*", "deep model*"]  
    **Transformed Concept List:** ["tur*", "languag intellig*", "ai*", "gener*", "statist*", "neural*", "pre-train*", "transform*", "corpora*", "nlp*", "in-context*", "bert*", "chatgpt*", "adapt*", "utiliz*"]  
    **Query:**  
    (abs:"languag model*" OR abs:"plm*" OR abs:"larg languag*" OR abs:"llm*" OR abs:"deep model*") AND (abs:"tur*" OR abs:"languag intellig*" OR abs:"ai*" OR abs:"gener*" OR abs:"statist*" OR abs:"neural*" OR abs:"pre-train*" OR abs:"transform*" OR abs:"corpora*" OR abs:"nlp*" OR abs:"in-context*" OR abs:"bert*" OR abs:"chatgpt*" OR abs:"adapt*" OR abs:"utiliz*")
    2. **Topic:** Quantum Computing  
    **Transformed Entity List:** ["quant comput*", "qubit*", "qc*", "quant devic*", "topolog comput*"]  
    **Transformed Concept List:** ["decoheren*", "entangl*", "error*", "topolog*", "anneal*", "photon*", "superconduct*", "algorithm*", "optim*", "verif*", "fault-toler*", "nois*", "cirquit*", "quant machin*", "measur*"]  
    **Query:**
    (abs:"quant comput*" OR abs:"qubit*" OR abs:"qc*" OR abs:"quant devic*" OR abs:"topolog comput*") AND (abs:"decoheren*" OR abs:"entangl*" OR abs:"error*" OR abs:"topolog*" OR abs:"anneal*" OR abs:"photon*" OR abs:"superconduct*" OR abs:"algorithm*" OR abs:"optim*" OR abs:"verif*" OR abs:"fault-toler*" OR abs:"nois*" OR abs:"cirquit*" OR abs:"quant machin*" OR abs:"measur*")
    ---

    ### **Now Generate the Query for This Topic:**
    Using the provided **Entity List** and **Concept List**, apply the following steps:
    1. **Ensure Entity List contains at least 5 items.** If fewer, supplement additional relevant terms.
    2. **Ensure Concept List contains 12-15 items.** If fewer, supplement additional relevant terms.
    3. **Convert all terms to their base form and append `*`.**
    4. **Construct the arXiv search query in the same format as the examples above.**
    5. **Return only the final query. Do not include explanations or additional text.**
    """

    # Initialize the OpenAI API client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    messages = [
        {"role": "system", "content": system_prompt_query},
        {"role": "user", "content": user_prompt_query}
    ]
    
    response = client.chat.completions.create(
        model="Qwen2.5-72B-Instruct",
        max_tokens=512,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=messages
    )
    
    output_query = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            output_query += chunk.choices[0].delta.content
    match = re.search(r'\(.*\)', output_query, re.DOTALL)

    if match:
        extracted_query = match.group(0)  # 保留匹配到的整个括号内容
    else:
        extracted_query = output_query.strip()  # 如果匹配失败，使用原始查询

    # 重新拼接 `submittedDate`
    updated_query = f"{extracted_query} AND submittedDate:[{start_date} TO {end_date}]"
    print('The response is :', updated_query)
    return updated_query.strip()


# Example usage:
if __name__ == "__main__":
    topic = "Quantum Computing"
    final_query = generate_arxiv_query_chain_of_thought(topic)
    print("\nFinal Query Returned:")
    print(final_query)