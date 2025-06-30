import json
import math
import os
import dotenv
from prompts import CONTENT_EVALUATION_PROMPT, OUTLINE_EVALUATION_PROMPT, CRITERIA, OUTLINE_STRUCTURE_PROMPT, REFERENCE_EVALUATION_PROMPT, OUTLINE_COVERAGE_PROMPT
from utils import build_outline_tree_from_levels, extract_and_save_outline_from_md, extract_references_from_md, extract_topic_from_path, getClient, generateResponse, robust_json_parse,fill_single_criterion_prompt
import logging

class Judge():
    def __init__(self):
        dotenv.load_dotenv()
        with open('judge.log', 'w') as log_file:
            log_file.truncate(0)
        self.client = getClient()
        # Configure logging
        logging.basicConfig(filename='judge.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    def judge(self, prompt):
        """
        :param prompt: str
        :return: str
        """
        response = generateResponse(self.client, prompt)
        logging.info(f"Response received: {response}")  # Log the response
        try:
            result = robust_json_parse(response)
            return result
        except Exception as e:
            logging.error(f"Error parsing JSON: {e}")  # Log the error
            print("Error parsing JSON:", e)
            return None
        
judge = Judge()

def extract_topic_from_path(md_path: str) -> str:
    """
    Extract topic from the grandparent folder name of the given path.
    """
    abs_path = os.path.abspath(md_path)
    topic = os.path.basename(os.path.dirname(os.path.dirname(abs_path)))
    return topic

def evaluate_outline_llm(outline_json_path: str) -> dict:
    criteria_name = "Outline"
    results = {}
    try:
        # 1. 读取 outline.json
        with open(outline_json_path, "r", encoding="utf-8") as f:
            outline_list = json.load(f)

        # 2. 格式化 outline 为字符串
        outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in outline_list])

        # 3. 用祖父文件夹名作为 topic
        topic = extract_topic_from_path(outline_json_path)

        # 4. 构建 prompt 并获取分数
        criterion = CRITERIA[criteria_name]
        prompt = fill_single_criterion_prompt(
            prompt_template=OUTLINE_EVALUATION_PROMPT,
            content=outline_str,
            topic=topic,
            criterion=criterion,
            criteria_name=criteria_name,
            type="outline"
        )
        score_dict = judge.judge(prompt)
        if not (isinstance(score_dict, dict) and criteria_name in score_dict):
            results[criteria_name] = 0
        else:
            results.update(score_dict)
    except Exception as e:
        results[criteria_name] = 0
    return results

def evaluate_outline_coverage(
    outline_json_path: str,
    standard_count: int = 10,
    ideal_section_count: int = 30,
    sigma: float = 15.0
) -> float:
    """
    评估大纲综合得分 Q'，融合模板完整度、创新丰富度和长度惩罚。

    Args:
        outline_json_path (str): 大纲JSON路径
        standard_count (int): 标准section总数 N
        ideal_section_count (int): 理想section总数 M*
        sigma (float): 惩罚宽度参数

    Returns:
        float: 综合得分 Q'
    """
    try:
        with open(outline_json_path, "r", encoding="utf-8") as f:
            outline_list = json.load(f)

        total_section_count = len(outline_list)  # M

        outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in outline_list])
        topic = extract_topic_from_path(outline_json_path)
        prompt = OUTLINE_COVERAGE_PROMPT.format(
            outline=outline_str,
            topic=topic,
        )
        response = judge.judge(prompt)
        matched_count = response.get("matched_count", 0)   # K

        K = matched_count
        N = standard_count
        M = total_section_count
        M_star = ideal_section_count
        U = max(M - K, 0)

        R = K / N if N > 0 else 0
        O = U / M if M > 0 else 0

        F_harmonic = 2 * R * O / (R + O) if (R + O) > 0 else 0

        L = math.exp(-((M - M_star) ** 2) / (2 * sigma ** 2)) if sigma > 0 else (1.0 if M == M_star else 0.0)

        Q_prime = F_harmonic * L

        return Q_prime

    except Exception as e:
        print("Error in evaluating outline coverage:", e)
        return 0.0

def evaluate_outline_structure(outline_json_path):
    """
    对 [level, title] 大纲结构评估层级合理性
    
    Returns:
        global_score: 全局结构合理性得分
        node_scores: 每个非叶节点的分数列表
    """
    with open(outline_json_path, "r", encoding="utf-8") as f:
        outline_list = json.load(f)
    node_objs, _ = build_outline_tree_from_levels(outline_list)
    non_leaf_nodes = [node for node in node_objs if node["children"]]
    node_scores = []
    for parent in non_leaf_nodes:
        children_list = "\n".join([
            f'  - Index: {child["index"]}, Title: {child["title"]}'
            for child in parent["children"]
        ])
        prompt = OUTLINE_STRUCTURE_PROMPT.format(
            parent_index=parent["index"],
            parent_title=parent["title"],
            children_list=children_list
        )
        # LLM调用（这里用模拟的judge函数代替）
        response = judge.judge(prompt)
        result = response.get("children", [])
        yes_count = sum(1 for child in result if str(child.get("is_included", "")).lower() == "yes")
        total = len(result)
        node_score = yes_count / total if total > 0 else 1.0  # 无子节点记满分（通常会被过滤掉）
        node_scores.append({
            "parent_index": parent["index"],
            "parent_title": parent["title"],
            "score": node_score
        })

    global_score = sum(x["score"] for x in node_scores) / len(node_scores) if node_scores else 1.0
    return global_score, node_scores

def evaluate_outline(md_path: str) -> dict:
    results = {}
    try:
        # 1. Extract outline from md
        extract_and_save_outline_from_md(md_path)
    except Exception as e:
        print("Error extracting outline:", e)
        return results
    try:
        # 2.LLM
        outline_json_path = os.path.join(os.path.dirname(md_path), "outline.json")
        outline_results = evaluate_outline_llm(outline_json_path)
        results.update(outline_results)
    except Exception as e:
        results["Outline"] = 0
    
    try:
        # 3. Coverage
        coverage_results = evaluate_outline_coverage(outline_json_path)
        results["Outline Coverage"] = coverage_results
    except Exception as e:
        print("Error in evaluating outline coverage:", e)
        results["Outline Coverage"] = 0
    
    try:
        # 4. Structure
        global_score, node_scores = evaluate_outline_structure(outline_json_path)
        results["Outline Structure"] = global_score
    except Exception as e:
        print("Error in evaluating outline structure:", e)
        results["Outline Structure"] = 0
    print("The score is: ", results)
    return results

def evaluate_content(md_path: str) -> dict:
    """
    Evaluate all specified content-related criteria and return a dict of all scores.
    If file reading or scoring fails, assign a score of 0 for that criterion.
    :param md_path: Markdown file path
    :return: dict {criteria_name: score, ...}
    """
    content_criteria = ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]
    results = {}

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content_str = f.read()
    except Exception as e:
        for criteria_name in content_criteria:
            results[criteria_name] = 0
        print("All content criteria scores:", results)
        return results

    topic = extract_topic_from_path(md_path)

    for criteria_name in content_criteria:
        criterion = CRITERIA[criteria_name]
        prompt = fill_single_criterion_prompt(
            prompt_template=CONTENT_EVALUATION_PROMPT,
            content=content_str,
            topic=topic,
            criterion=criterion,
            criteria_name=criteria_name,
            type= "content"
        )
        try:
            score_dict = judge.judge(prompt)
            if not (isinstance(score_dict, dict) and criteria_name in score_dict):
                results[criteria_name] = 0
            else:
                results.update(score_dict)
        except Exception as e:
            results[criteria_name] = 0

    print("All content criteria scores:", results)
    return results

def evaluate_reference(md_path: str) -> dict:
    """
    Extract references from the given Markdown file, call LLM to evaluate relevance,
    and return {"Reference": score}. If extraction or scoring fails, assign a score of 0.
    :param md_path: Markdown file path
    :return: dict, key is "Reference", value is the score
    """
    results = {}
    criteria_name = "Reference"
    try:
        references = extract_references_from_md(md_path)
        if not references:
            results[criteria_name] = 0
            print("No references found.")
            return results

        topic = extract_topic_from_path(md_path)
        criterion = CRITERIA[criteria_name]
        references_str = "\n".join(references)
        prompt = fill_single_criterion_prompt(
            prompt_template=REFERENCE_EVALUATION_PROMPT,
            content=references_str,
            topic=topic,
            criterion=criterion,
            criteria_name=criteria_name,
            type="reference"
        )
        try:
            score_dict = judge.judge(prompt)
            if not (isinstance(score_dict, dict) and criteria_name in score_dict):
                results[criteria_name] = 0
            else:
                results.update(score_dict)
        except Exception:
            print("Error in scoring references.")
            results[criteria_name] = 0
    except Exception:
        print("Error in extracting references.")
        results[criteria_name] = 0
    print("Reference evaluation score:", results)
    return results

if __name__ == "__main__":
    # 测试代码
    #md_path = "surveys/cs/3D Gaussian Splatting Techniques/AutoSurvey/3D Gaussian Splatting Techniques.md"  # 替换为实际的文件路径
    md_path = "surveys\cs\Agent-based Modeling and Simulation using Large Language Models\AutoSurvey\Agent-based Modeling and Simulation using Large Language Models.md"
    json_path = os.path.join(os.path.dirname(md_path), "outline.json")
    evaluate_outline(md_path)
    
    # evaluate_content(md_path)
    # evaluate_reference(md_path)
    # print(evaluate_outline_coverage(json_path))



