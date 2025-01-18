import os
from openai import OpenAI
from datetime import datetime, timedelta

def generate_query_qwen(topic):
    # 获取当前日期并计算最近三年范围
    today = datetime.now()
    three_years_ago = today - timedelta(days=3*365)  # 简单近似，忽略闰年
    start_date = three_years_ago.strftime('%Y%m%d')  # 格式化为 arXiv 格式
    end_date = today.strftime('%Y%m%d')  # 当前日期

    # Prompt that provides instructions to the assistant
    system_prompt = f'''
    You are a skilled research assistant specializing in crafting broad but effective search queries for the arXiv scientific paper repository. Broad first.
    '''

    # User prompt that specifies the task and guidelines
    user_prompt = f'''
Task: Craft an effective and flexible search query tailored for the arXiv database, specifically designed to retrieve research papers pertaining to the following topic:

Topic: {topic}

Guidelines:
1. The query must contain **three main parts**, connected by the logical operator `AND`:
    - **Part 1**: Extracts the 3 most likely entities or concepts related to the topic along with their synonyms, abbreviations, and alternative expressions. This part must:
        - Include synonyms, abbreviations, and alternative terms (e.g., "LLM" for "large language model").
        - Use wildcards (`*`) where applicable to capture variations (e.g., `model*` matches `model`, `models`, etc.).
        - Connect keywords using the `OR` operator to broaden the scope.
        - Include both `ti:` (title) and `abs:` (abstract) fields.
    - **Part 2**: Describes fields and applications relevant to the topic, using **3 synonyms or closely related terms** restricted to the `abs:` field only:
        - The terms must directly relate to the field or application described in the topic.
        - Use the `OR` operator to connect terms.
    - **Part 3**: Represents actions, methods, or processes relevant to the topic, using **3 verbs or action words**, restricted to the `abs:` field only:
        - Use wildcards (`*`) whenever applicable to capture variations (e.g., `detect*` matches `detect`, `detecting`, etc.).
        - Connect keywords using the `OR` operator.

2. Use wildcards (`*`) in all parts where appropriate to capture variations (e.g., plural forms or word stems).
3. Ensure the query structure is simple and adheres to arXiv's search syntax, with logical operators (`AND`, `OR`) and fields (`ti:` for title, `abs:` for abstract).
4. Avoid excessive repetition and overloading the query with too many terms.
5. Prioritize flexibility to maximize the number of relevant papers retrieved.
6. Each term should not exceed 2 words to maintain query simplicity and effectiveness.
7. Restrict the query to papers **submitted within the last three years** using the date range `submittedDate:[{start_date} TO {end_date}]`.

Example Outputs:

| Topic                                      | Query                                                                                                                                                                  |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Automating Literature Review Generation with LLM | (ti:"LLM*" OR abs:"LLM*" OR ti:"large language model*" OR abs:"large language model*" OR ti:"language model*" OR abs:"language model*") AND (abs:"literature review*" OR "review generation*" OR "survey generation*") AND (abs:"automate*" OR "generate*" OR "summariz*") AND submittedDate:[{start_date} TO {end_date}] |
| Graph Neural Networks for Social Networks  | (ti:"graph neural*" OR abs:"graph neural*" OR ti:"GNN*" OR abs:"GNN*" OR ti:"graph*" OR abs:"graph*") AND (abs:"social network*" OR "social graph*" OR "relation*") AND (abs:"detect*" OR "classify*" OR "analyz*") AND submittedDate:[{start_date} TO {end_date}] |
| Reinforcement Learning for Robotics        | (ti:"reinforcement learning" OR abs:"reinforcement learning" OR ti:"RL" OR abs:"RL" OR ti:"reinforce*" OR abs:"reinforce*") AND (abs:"robot*" OR "agent*" OR "robotic*") AND (abs:"optim*" OR "control*" OR "learn*") AND submittedDate:[{start_date} TO {end_date}] |
| Explainable AI in Cybersecurity            | (ti:"explainable AI" OR abs:"explainable AI" OR ti:"XAI" OR abs:"XAI" OR ti:"interpretable model*" OR abs:"interpretable model*") AND (abs:"cybersecurity*" OR "security*" OR "secur*") AND (abs:"explain*" OR "detect*" OR "analyz*") AND submittedDate:[{start_date} TO {end_date}] |
| Multimodal Learning for Vision and Language| (ti:"multimodal*" OR abs:"multimodal*" OR ti:"multimodal model*" OR abs:"multimodal model*" OR ti:"cross-modal*" OR abs:"cross-modal*") AND (abs:"vision*" OR "imag*" OR "video*") AND (abs:"recogniz*" OR "process*" OR "generate*") AND submittedDate:[{start_date} TO {end_date}] |

Output format:
* Provide the arxiv query only. Do not include any additional explanations or commentary.
    '''

    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_prompt},
        ]
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model="Qwen2.5-72B-Instruct",
        max_tokens=512,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages= messages
    )
    # Stream the response to console
    text = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    print('The response is :', text)
    return text.strip()