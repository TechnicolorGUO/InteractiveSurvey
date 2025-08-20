import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --------------------------------------------------------------------------------
# PROMPTS & CLIENT UTILS
# --------------------------------------------------------------------------------
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

def getQwenClient(): 
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def generateResponse(client, prompt):
    chat_response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        max_tokens=32768,
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

# --------------------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    client = getQwenClient()

    category_folders = [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Statistics",
        "Electrical Engineering and Systems Science",
        "Quantitative Biology",
        "Quantitative Finance",
        "Economics"
    ]

    evaluation_results = {}

    for category in category_folders:
        if not os.path.isdir(category):
            # If the folder doesn't exist, skip
            print(f"Skipping: '{category}' - directory not found.")
            continue

        # Initialize a dict for this category in the results
        evaluation_results[category] = {}

        # For each .md file found in this category folder
        for filename in os.listdir(category):
            # We only want .md files that follow the naming pattern "survey_{topic}.md"
            if filename.lower().endswith(".md") and filename.startswith("survey_"):
                # Extract the topic from the filename
                # e.g., "survey_LLM for In-Context Learning.md" -> "LLM for In-Context Learning"
                topic = filename[len("survey_") : -len(".md")]

                md_file_path = os.path.join(category, filename)
                if not os.path.isfile(md_file_path):
                    continue

                # Read the content of the survey file
                with open(md_file_path, "r", encoding="utf-8") as f:
                    survey_content = f.read()

                # Evaluate
                try:
                    coverage_score = evaluate_coverage(topic, survey_content, client)
                    structure_score = evaluate_structure(topic, survey_content, client)
                    relevance_score = evaluate_relevance(topic, survey_content, client)

                    # Store in nested dictionary: results[category][topic] = ...
                    evaluation_results[category][topic] = {
                        "coverage": coverage_score,
                        "structure": structure_score,
                        "relevance": relevance_score
                    }

                    print(f"Evaluated: {category} / {topic}")
                except Exception as e:
                    print(f"Error evaluating '{category} / {topic}': {e}")

    # Write everything to a single JSON file
    output_file = "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Results saved to: {output_file}")
