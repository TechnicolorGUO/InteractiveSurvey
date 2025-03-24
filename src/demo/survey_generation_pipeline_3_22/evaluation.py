import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def getQwenClient(): 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key = openai_api_key,
        base_url = openai_api_base,
    )
    return client
def generateResponse(client, prompt):
    chat_response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        max_tokens=128,
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

COVERAGE_PROMPT = '''
Here is an academic survey about the topic "[TOPIC]":
---
[SURVEY]
---

<instruction>
Please evaluate this survey about the topic "[TOPIC]" based on the criterion provided below and give a score from 1 to 5 according to the score description:
---
Criterion Description: Coverage assesses the extent to which the survey encapsulates all relevant aspects of the topic.
---
Score 1 Description: The survey has very limited coverage, only touching on a small portion of the topic and lacking discussion on key areas.
Score 2 Description: The survey covers some parts of the topic but has noticeable omissions, with significant areas either underrepresented or missing.
Score 3 Description: The survey is generally comprehensive but still misses a few key points.
Score 4 Description: The survey covers most key areas comprehensively, with only very minor topics left out.
Score 5 Description: The survey comprehensively covers all key and peripheral topics, providing detailed discussions and extensive information.
---
Return the score without any other information:
'''

STRUCTURE_PROMPT = '''
Here is an academic survey about the topic "[TOPIC]":
---
[SURVEY]
---

<instruction>
Please evaluate this survey about the topic "[TOPIC]" based on the criterion provided below and give a score from 1 to 5 according to the score description:
---
Criterion Description: Structure evaluates the logical organization and coherence of sections and subsections.
---
Score 1 Description: The survey lacks logic, with no clear connections between sections.
Score 2 Description: The survey has weak logical flow with some disordered content.
Score 3 Description: The survey has a generally reasonable logical structure.
Score 4 Description: The survey has good logical consistency, with content well arranged.
Score 5 Description: The survey is tightly structured and logically clear.
---
Return the score without any other information:
'''

RELEVANCE_PROMPT = '''
Here is an academic survey about the topic "[TOPIC]":
---
[SURVEY]
---

<instruction>
Please evaluate this survey about the topic "[TOPIC]" based on the criterion provided below and give a score from 1 to 5 according to the score description:
---
Criterion Description: Relevance measures how well the content aligns with the research topic.
---
Score 1 Description: The content is outdated or unrelated to the field.
Score 2 Description: The survey is somewhat on topic but with several digressions.
Score 3 Description: The survey is generally on topic, despite a few unrelated details.
Score 4 Description: The survey is mostly on topic and focused.
Score 5 Description: The survey is exceptionally focused and entirely on topic.
---
Return the score without any other information:
'''

def evaluate_survey(topic, survey_content, client, prompt_template):
    prompt = prompt_template.replace("[TOPIC]", topic).replace("[SURVEY]", survey_content)
    response = generateResponse(client, prompt)
    return response.strip()

def evaluate_coverage(topic, survey_content, client):
    return evaluate_survey(topic, survey_content, client, COVERAGE_PROMPT)

def evaluate_structure(topic, survey_content, client):
    return evaluate_survey(topic, survey_content, client, STRUCTURE_PROMPT)

def evaluate_relevance(topic, survey_content, client):
    return evaluate_survey(topic, survey_content, client, RELEVANCE_PROMPT)

if __name__ == "__main__":
    client = getQwenClient()
    result_dir = "./result_3_22"  # 当前 evaluation.py 同级目录下的 result 文件夹

    evaluation_results = {}  # 存储所有 topic 的评分结果

    # 遍历 result 文件夹中的所有子文件夹
    for topic_dir in os.listdir(result_dir):
        topic_path = os.path.join(result_dir, topic_dir)
        if os.path.isdir(topic_path):
            topic = topic_dir  # 子文件夹名称作为 topic，例如 "LLM for In-Context Learning"
            # md 文件命名规则为 survey_{topic}.md
            md_filename = f"survey_{topic}.md"
            md_file = os.path.join(topic_path, md_filename)
            if os.path.exists(md_file):
                with open(md_file, "r", encoding="utf-8") as f:
                    survey_content = f.read()
                try:
                    coverage_score = evaluate_coverage(topic, survey_content, client)
                    structure_score = evaluate_structure(topic, survey_content, client)
                    relevance_score = evaluate_relevance(topic, survey_content, client)
                    evaluation_results[topic] = {
                        "coverage": coverage_score,
                        "structure": structure_score,
                        "relevance": relevance_score
                    }
                    print(f"Processed topic: {topic}")
                except Exception as e:
                    print(f"Error processing topic '{topic}': {e}")
            else:
                print(f"MD file not found for topic: {topic} at path: {md_file}")

    # 将所有评分结果存储到 evaluation_results.json 中
    output_file = "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_file}")

# if __name__ == "__main__":
#     client = getQwenClient()
    
#     topic = "LLM for In-Context Learning"
#     md_file = "./result/LLM for In-Context Learning/survey_LLM for In-Context Learning.md"
    
#     with open(md_file, "r", encoding="utf-8") as f:
#         survey_content = f.read()
    
#     coverage_score = evaluate_coverage(topic, survey_content, client)
#     structure_score = evaluate_structure(topic, survey_content, client)
#     relevance_score = evaluate_relevance(topic, survey_content, client)
    
#     print(f"Coverage Score: {coverage_score}")
#     print(f"Structure Score: {structure_score}")
#     print(f"Relevance Score: {relevance_score}")
