import os
from openai import OpenAI
from datetime import datetime, timedelta
import re
def generate_query_qwen(topic):
    # 获取当前日期并计算最近三年范围
    today = datetime.now()
    five_years_ago = today - timedelta(days=5*365)  # 简单近似，忽略闰年
    start_date = five_years_ago.strftime('%Y%m%d')  # 格式化为 arXiv 格式
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
    1. The query must contain **four main parts**, connected by the logical operator `AND`:
        - **Part 1**: Extracts the 3 most likely entities or concepts related to the topic along with their synonyms, abbreviations, and alternative expressions, restricted to the `abs:` field only:
            - Include synonyms, abbreviations, and alternative terms (e.g., "LLM" for "large language model").
            - Use wildcards (`*`) where applicable to capture variations (e.g., `model*` matches `model`, `models`, etc.).
            - Connect keywords using the `OR` operator to broaden the scope.
        - **Part 2**: Describes fields and applications relevant to the topic, using **3 synonyms or closely related terms** restricted to the `abs:` field only:
            - The terms must directly relate to the field or application described in the topic.
            - Use the `OR` operator to connect terms.
        - **Part 3**: Represents actions, methods, or processes relevant to the topic, using **3 verbs or action words**, restricted to the `abs:` field only:
            - Use wildcards (`*`) whenever applicable to capture variations (e.g., `detect*` matches `detect`, `detecting`, etc.).
            - Connect keywords using the `OR` operator.
        - **Part 4**: Restricts the query to two **arXiv categories (large categories)** that are most relevant to the topic:
            - Use `cat:` to specify the two categories (e.g., `cs` for Computer Science, `stat` for Statistics).
            - Combine categories using the `OR` operator to broaden the scope.

    2. Use wildcards (`*`) in all parts where appropriate to capture variations (e.g., plural forms or word stems).
    3. Ensure the query structure is simple and adheres to arXiv's search syntax, with logical operators (`AND`, `OR`) and fields (`ti:` for title, `abs:` for abstract, `cat:` for categories).
    4. Avoid excessive repetition and overloading the query with too many terms.
    5. Prioritize flexibility to maximize the number of relevant papers retrieved.
    6. Each term should not exceed 2 words to maintain query simplicity and effectiveness.
    7. Restrict the query to papers **submitted within the last three years** using the date range `submittedDate:[{start_date} TO {end_date}]`.

    Example Outputs:

    | Topic                                      | Query                                                                                                                                                                                                                                                                                          |
    | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | Automating Literature Review Generation with LLM | (abs:"LLM*" OR abs:"large language model*" OR abs:"language model*") AND (abs:"literature review*" OR "review generation*" OR "survey generation*") AND (abs:"automate*" OR "generate*" OR "summariz*") AND (cat:cs OR cat:stat) AND submittedDate:[{start_date} TO {end_date}] |
    | Graph Neural Networks for Social Networks  | (abs:"graph neural*" OR abs:"GNN*" OR abs:"graph*") AND (abs:"social network*" OR "social graph*" OR "relation*") AND (abs:"detect*" OR "classify*" OR "analyz*") AND (cat:cs OR cat:stat) AND submittedDate:[{start_date} TO {end_date}] |
    | Reinforcement Learning for Robotics        | (abs:"reinforcement learning" OR abs:"RL" OR abs:"reinforce*") AND (abs:"robot*" OR "agent*" OR "robotic*") AND (abs:"optim*" OR "control*" OR "learn*") AND (cat:cs OR cat:stat) AND submittedDate:[{start_date} TO {end_date}] |
    | Explainable AI in Cybersecurity            | (abs:"explainable AI" OR abs:"XAI" OR abs:"interpretable model*") AND (abs:"cybersecurity*" OR "security*" OR "secur*") AND (abs:"explain*" OR "detect*" OR "analyz*") AND (cat:cs OR cat:stat) AND submittedDate:[{start_date} TO {end_date}] |
    | Multimodal Learning for Vision and Language| (abs:"multimodal*" OR abs:"multimodal model*" OR abs:"cross-modal*") AND (abs:"vision*" OR "imag*" OR "video*") AND (abs:"recogniz*" OR "process*" OR "generate*") AND (cat:cs OR cat:stat) AND submittedDate:[{start_date} TO {end_date}] |

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

    category_map = {
        "cs": [
            "cs.AI", "cs.CL", "cs.CC", "cs.CE", "cs.CG", "cs.GT", "cs.CV", "cs.CY",
            "cs.CR", "cs.DS", "cs.DB", "cs.DL", "cs.DM", "cs.DC", "cs.ET", "cs.FL",
            "cs.GL", "cs.GR", "cs.HC", "cs.IR", "cs.IT", "cs.LO", "cs.LG", "cs.MA",
            "cs.MM", "cs.NI", "cs.NE", "cs.NA", "cs.OS", "cs.OH", "cs.PF", "cs.PL",
            "cs.RO", "cs.SI", "cs.SE", "cs.SD", "cs.SC"
        ],
        "stat": [
            "stat.AP", "stat.CO", "stat.ML", "stat.ME", "stat.OT", "stat.TH"
        ],
        "physics": [
            "astro-ph.GA", "astro-ph.CO", "astro-ph.EP", "astro-ph.HE", "astro-ph.IM", "astro-ph.SR",
            "cond-mat.dis-nn", "cond-mat.mtrl-sci", "cond-mat.mes-hall", "cond-mat.other",
            "cond-mat.quant-gas", "cond-mat.soft", "cond-mat.stat-mech", "cond-mat.str-el",
            "cond-mat.supr-con", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph",
            "nlin.AO", "nlin.CG", "nlin.CD", "nlin.SI", "nlin.PS", "nucl-ex", "nucl-th",
            "physics.acc-ph", "physics.app-ph", "physics.ao-ph", "physics.atom-ph", "physics.bio-ph",
            "physics.chem-ph", "physics.class-ph", "physics.comp-ph", "physics.data-an",
            "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph", "physics.hist-ph",
            "physics.ins-det", "physics.med-ph", "physics.optics", "physics.ed-ph",
            "physics.soc-ph", "physics.plasm-ph", "physics.pop-ph", "physics.space-ph",
            "quant-ph"
        ],
        "math": [
            "math.AG", "math.AT", "math.AP", "math.CT", "math.CA", "math.CO", "math.AC",
            "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GT",
            "math.GR", "math.HO", "math.IT", "math.KT", "math.LO", "math.MP", "math.MG",
            "math.NT", "math.NA", "math.OA", "math.OC", "math.PR", "math.QA", "math.RT",
            "math.RA", "math.SP", "math.ST", "math.SG"
        ],
        "q-bio": [
            "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT",
            "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"
        ],
        "q-fin": [
            "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR",
            "q-fin.RM", "q-fin.ST", "q-fin.TR"
        ],
        "eess": [
            "eess.AS", "eess.IV", "eess.SP", "eess.SY"
        ],
        "econ": [
            "econ.EM", "econ.GN", "econ.TH"
        ]
    }
    def replace_categories(query, category_map):
        # 遍历每个大类
        for category, subcategories in category_map.items():
            # 构造子类别的替换字符串
            replacement = f"(cat:{' OR cat:'.join(subcategories)})"
            # 使用正则表达式找到并替换对应的大类
            # 注意 \b 确保匹配完整的词边界，避免部分匹配错误
            query = re.sub(rf"\bcat:{category}\b", replacement, query)
        return query

    updated_query = replace_categories(text, category_map)
    print('The response is :', updated_query)
    return updated_query.strip()