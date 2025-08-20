import os
from openai import OpenAI
from datetime import datetime, timedelta
import re

def _strip_think_blocks(text):
    """Remove <think>...</think> or <thinking>...</thinking> blocks from model outputs."""
    if not text:
        return text
    cleaned = re.sub(r'<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'<\s*thinking\s*>[\s\S]*?<\s*/\s*thinking\s*>', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

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
        model=os.environ.get("MODEL"),
        max_tokens=32768,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=messages_abstract,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    
    abstract_text = ""
    for chunk in abstract_response:
        if chunk.choices[0].delta.content:
            abstract_text += chunk.choices[0].delta.content
    abstract_text = _strip_think_blocks(abstract_text)
    # print("The abstract is:", abstract_text)

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
    1. **Entity list**:  5 Entities that are synonymous or closely related to the given topic (nouns only). These should be concise (no more than two words) and simplified to their root forms (e.g., removing suffixes like "-ing", "-ed") such as llm for large language model.  
    2. **Concept list**: Core concepts from the abstract that are highly relevant to the topic. These should also be concise (no more than two words) and in their simplest form, one single word is preferred unless the term is inseparable.  

    Ensure that your response follows this exact format:
    Entity list: [entity1, entity2, entity3, entity4, entity5]
    Concept list: [concept1, concept2, concept3, ...concept n]
    Do not include any explanations or additional text.  

    ### **Example**  
    #### **Input:**  
    Topic: Large Language Models  
    Abstract: Ever since the Turing Test was proposed in the 1950s, humans have explored the mastering of language intelligence by machine. Language is essentially a complex, intricate system of human expressions governed by grammatical rules. It poses a significant challenge to develop capable artificial intelligence (AI) algorithms for comprehending and grasping a language. As a major approach, language modeling has been widely studied for language understanding and generation in the past two decades, evolving from statistical language models to neural language models. Recently, pre-trained language models (PLMs) have been proposed by pretraining Transformer models over large-scale corpora, showing strong capabilities in solving various natural language processing (NLP) tasks. Since the researchers have found that model scaling can lead to an improved model capacity, they further investigate the scaling effect by increasing the parameter scale to an even larger size. Interestingly, when the parameter scale exceeds a certain level, these enlarged language models not only achieve a significant performance improvement, but also exhibit some special abilities (e.g., in-context learning) that are not present in small-scale language models (e.g., BERT). To discriminate the language models in different parameter scales, the research community has coined the term large language models (LLM) for the PLMs of significant size (e.g., containing tens or hundreds of billions of parameters). Recently, the research on LLMs has been largely advanced by both academia and industry, and a remarkable progress is the launch of ChatGPT (a powerful AI chatbot developed based on LLMs), which has attracted widespread attention from society. The technical evolution of LLMs has been making an important impact on the entire AI community, which would revolutionize the way how we develop and use AI algorithms. Considering this rapid technical progress, in this survey, we review the recent advances of LLMs by introducing the background, key findings, and mainstream techniques. In particular, we focus on four major aspects of LLMs, namely pre-training, adaptation tuning, utilization, and capacity evaluation. Furthermore, we also summarize the available resources for developing LLMs and discuss the remaining issues for future directions. This survey provides an up-to-date review of the literature on LLMs, which can be a useful resource for both researchers and engineers.  

    #### **Expected Output:**
    "entity list": ["language model", "plm", "large language", "llm", "llms"]  
    "concept list": ["turing", "language intelligence", "ai", "generation", "statistical", "neural", "pre-train", "transformer", "corpora", "nlp", "in-context", "bert", "chatgpt", "adaptation", "utilization"]
    Make sure to strictly follow this format in your response.
    """

    user_prompt_abstract = f"""
    Topic: {topic}  
    Abstract: {abstract_text}  

    Based on the given topic and abstract, extract the following:  
    1. A **list of 5 most key entities (nouns)** that are synonymous or closely related to the topic. Keep each entity under two words and in its simplest form.  
    2. A **list of core concepts (terms) as many as possible** from the abstract that are highly relevant to the topic. Keep each concept under two words and in its simplest form.     
    """

    messages_abstract = [
        {"role": "system", "content": system_prompt_abstract},
        {"role": "user", "content": user_prompt_abstract}
    ]
    
    entity_response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        max_tokens=32768,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=messages_abstract,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    
    entity_list = ""
    for chunk in entity_response:
        if chunk.choices[0].delta.content:
            entity_list += chunk.choices[0].delta.content
    entity_list = _strip_think_blocks(entity_list)
    # print("The entity lists are:", entity_list)

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
    You are a research assistant specializing in constructing effective arXiv search queries. Your task is to generate a structured search query using **pre-extracted entity and concept lists** from a given abstract and the topic. Follow these instructions exactly:

    1. **Input Data:**
    - **Entity List:** A list of entities that are synonymous or closely related to the given topic.
    - **Concept List:** A list of core concepts from the abstract that are highly relevant to the topic.

    2. **Ensure Minimum Keyword Count:**
    - **Entity List** must contain at least **3** nouns of entities. If there are fewer, intelligently supplement additional relevant terms, ensuring that entities are synonyms or closely related to the key entity in the topic (e.g., "LLM" for "Large Language Model").
    - **Concept List** must contain **12-15** domain-specific terms. If there are fewer, intelligently supplement additional relevant terms. Avoid broad terms like "combine" or "introduce."

    3. **Standardize Formatting:**
    - Convert all terms to their **base form** without adding any wildcard (`*`).
    - All terms must be **in lowercase**.

    4. **Construct the Final Query:**
    - The query must follow this exact structure:
        ```
        (abs:"<Term1>" AND abs:"<Term2>") AND
        (abs:"<Entity1>" OR abs:"<Entity2>" OR abs:"<Entity3>" OR abs:"<Entity4>" OR abs:"<Entity5>") AND 
        (abs:"<Concept1>" OR abs:"<Concept2>" OR ... OR abs:"<Concept12>")
        ```
    - **Terms are 2 or 3 keywords or phrases extracted from the topic that you think **must** occur in the abstract of the searching results and are grouped together using `AND` in the first part.** (most important)
    - **Entities are grouped together using `OR` in the second part.**
    - **Concepts are grouped together using `OR` in the third part.**
    - **The two groups are combined using `AND`.**
    - **For compound words with hyphens (e.g., "in-context"), replace `-` with a space, resulting in `"in context"`.**
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
    - **Ensure the key terms in the topic are included**.
    - **Ensure at least 5 entities** (if fewer, supplement additional relevant terms).
    - **Ensure 12-15 concepts** (if fewer, supplement additional relevant terms).
    - **Convert all terms to lowercase.**
    - **For compound words with hyphens (e.g., "in-context"), replace `-` with a space, resulting in `"in context"`**.
    - **Output only the final query with no extra text.**

    ### **Example Query Format:**

    1. **Topic:** Large Language Models in Recommendation Systems  
    **Transformed Entity List:** ["language model", "plm", "large language", "llm", "deep model"]  
    **Transformed Concept List:** ["tur", "language intelligence", "ai", "generation", "statistical", "neural", "pretraining", "transformer", "corpora", "nlp", "in context", "bert", "chatgpt", "adaptation", "utilization"]  
    **Query:**  
    (abs:"large language model" AND abs:"recommendation") AND (abs:"language model" OR abs:"plm" OR abs:"large language" OR abs:"llm" OR abs:"deep model") AND (abs:"tur" OR abs:"language intelligence" OR abs:"ai" OR abs:"generation" OR abs:"statistical" OR abs:"neural" OR abs:"pretraining" OR abs:"transformer" OR abs:"corpora" OR abs:"nlp" OR abs:"in context" OR abs:"bert" OR abs:"chatgpt" OR abs:"adaptation" OR abs:"utilization")

    2. **Topic:** Quantum Computing in Physics  
    **Transformed Entity List:** ["quantum computing", "qubit", "qc", "quantum device", "topological computing"]  
    **Transformed Concept List:** ["decoherence", "entanglement", "error", "topology", "annealing", "photon", "superconducting", "algorithm", "optimization", "verification", "fault tolerance", "noise", "circuit", "quantum machine", "measurement"]  
    **Query:**  
    (abs:"quantum computing" AND abs:"physics") AND (abs:"quantum computing" OR abs:"qubit" OR abs:"qc" OR abs:"quantum device" OR abs:"topological computing") AND (abs:"decoherence" OR abs:"entanglement" OR abs:"error" OR abs:"topology" OR abs:"annealing" OR abs:"photon" OR abs:"superconducting" OR abs:"algorithm" OR abs:"optimization" OR abs:"verification" OR abs:"fault tolerance" OR abs:"noise" OR abs:"circuit" OR abs:"quantum machine" OR abs:"measurement")

    ---

    ### **Now Generate the Query for This Topic:**
    **Topic:** {topic}  
    Using the provided **Entity List** and **Concept List**, apply the following steps:
    1. **Ensure Entity List contains at least 5 items.** If fewer, supplement additional relevant terms.
    2. **Ensure Concept List contains 12-15 items.** If fewer, supplement additional relevant terms.
    3. **Convert all terms to lowercase.**
    4. **For compound words with hyphens (`-`), replace `-` with a space, e.g., `"in-context"` → `"in context"`.**
    5. **Construct the arXiv search query in the same format as the examples above.**
    6. **Return only the final query. Do not include explanations or additional text.**
    All the terms in query should not exceed 2 words!
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
    
    output_query = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            output_query += chunk.choices[0].delta.content
    output_query = _strip_think_blocks(output_query)
    match = re.search(r'\(.*\)', output_query, re.DOTALL)

    if match:
        extracted_query = match.group(0)  # 保留匹配到的整个括号内容
    else:
        extracted_query = output_query.strip()  # 如果匹配失败，使用原始查询

    # 重新拼接 `submittedDate`
    # updated_query = f"{extracted_query} AND submittedDate:[{start_date} TO {end_date}]"
    updated_query = f"{extracted_query}"
    print('The response is :', updated_query)
    return updated_query.strip()

def generate_generic_query_qwen(original_query, topic):
    """
    Transforms an overly strict arXiv query into a simplified, more generic version.
    
    The new query must be in the format:
        (abs:"<GenericTerm1>" AND abs:"<GenericTerm2>") OR (abs:"<GenericTerm3>" AND abs:"<GenericTerm4>")
        
    Here, <GenericTerm1> and <GenericTerm2> represent two generic and common keywords,
    while <GenericTerm3> and <GenericTerm4> are synonyms or closely related terms to the first two.
    related to the given topic. If the terms in the original query are too strict,
    replace them with broader terms that improve matching against arXiv articles.
    
    Parameters:
        original_query (str): The output query from generate_query_qwen() which is too strict.
        topic (str): The research topic.
    
    Returns:
        str: The simplified arXiv query.
    """
    
    system_prompt = """
    You are a research assistant specializing in constructing effective and broad arXiv search queries.
    Your job is to transform an overly strict query into a simplified, generic one.
    
    Instructions:
    1. Input:
       - A strict query that might be too specific.
       - A topic which the query intends to capture.
    
    2. Requirements:
       - Create a new query that only has the structure:
         (abs:"<GenericTerm1>" AND abs:"<GenericTerm2>") OR (abs:"<GenericTerm3>" AND abs:"<GenericTerm4>")
       - Replace <GenericTerm1> and <GenericTerm2> with two generic and common keywords for the topic.
       - Replace <GenericTerm3> and <GenericTerm4> with the synonyms or closely related terms to the <GenericTerm1> and <GenericTerm2>.
       - If the terms from the original query are too narrow, modify them to more broadly represent the given topic.
       - All keywords must be in lowercase and in their base form.
       - Each term should be one or two words.
    
    3. Output:
       - Return only the final query in the exact format with no extra explanations.
    """
    
    user_prompt = f"""
    Original Query: {original_query}
    Topic: {topic}
    
    The original query may be too strict and fails to match a broad range of arXiv articles.
    Please generate a new query in the format:
        (abs:"<GenericTerm1>" AND abs:"<GenericTerm2>") OR (abs:"<GenericTerm3>" AND abs:"<GenericTerm4>")
    Replace <GenericTerm1> and <GenericTerm2> with more generic and commonly used terms that represent the topic.
    Output only the final query.
    """
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    
    # Initialize the OpenAI API client (assuming a similar interface as before)
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = client.chat.completions.create(
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
    
    output_query = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            output_query += chunk.choices[0].delta.content
    output_query = _strip_think_blocks(output_query)
    # Use regex to extract the new simplified query in the exact required format
    match = re.search(r'\(.*\)', output_query, re.DOTALL)
    if match:
        extracted_query = match.group(0)
    else:
        extracted_query = output_query.strip()
    
    print('The response is :', extracted_query)
    return extracted_query.strip()
