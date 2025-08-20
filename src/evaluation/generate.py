import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def getQwenClient():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def generateResponse(client, prompt):
    """
    Generates a response from Qwen 2.5 72B API using a larger max_tokens value
    to produce a long-form output.
    """
    chat_response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        max_tokens=32768,
        temperature=0.5,
        stop="<|im_end|>",
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    # Stream the response to gather the entire text
    text = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    return text

# Dictionary of categories and topics
arxiv_topics = {
    "Computer Science": [
        "LLM for In-Context Learning",
        "GANs in Computer Vision",
        "Reinforcement Learning for Autonomous Driving",
        "Self-Supervised Learning in NLP",
        "Quantum Computing for Machine Learning",
        "AI for Code Generation and Program Analysis",
        "Transformer Architectures for Natural Language Processing",
        "Federated Learning for Data Privacy",
        "Graph Neural Networks in Social Network Analysis",
        "Explainable AI for Decision Support Systems"
    ],
    "Mathematics": [
        "Graph Theory for Social Network Analysis",
        "Optimal Transport Theory in Machine Learning",
        "Nonlinear Dynamics in Chaotic Systems"
    ],
    "Physics": [
        "Physical Implementation of Quantum Computing",
        "Advances in Gravitational Wave Detection",
        "Topological Properties of Bose-Einstein Condensates",
        "Supersymmetry Theory in High Energy Physics",
        "AI Methods in Programmable Physics Simulations",
        "Dark Matter Modeling in Cosmology",
        "Neutrino Oscillations in Particle Physics",
        "Topological Insulators in Solid State Physics"
    ],
    "Statistics": [
        "Uncertainty Estimation in Deep Learning",
        "Statistical Physics in Random Matrix Theory",
        "Dimensionality Reduction in High-Dimensional Data",
        "Bayesian Optimization in Machine Learning",
        "Causal Inference Methods in Epidemiology",
        "High-Dimensional Data Analysis in Machine Learning",
        "Gaussian Process Regression in Robotics",
        "Statistical Learning Theory and Generalization Error Analysis"
    ],
    "Electrical Engineering and Systems Science": [
        "AI Optimization for 5G and 6G Networks",
        "Optimization of Quantum Sensors",
        "Low Power Design Methods in Electronics",
        "Quantum Dot Technologies in Display Systems"
    ],
    "Quantitative Biology": [
        "Computational Modeling in Neuroscience",
        "Single-Cell RNA Sequencing in Cancer Research"
    ],
    "Quantitative Finance": [
        "Reinforcement Learning in Algorithmic Trading",
        "Machine Learning for Credit Scoring",
        "Cryptocurrency Market Price Prediction",
        "Blockchain Technologies for Financial Services"
    ],
    "Economics": [
        "Modeling Climate Change Economics"
    ]
}

def create_directories():
    """
    Create one folder for each of the 8 categories.
    """
    for category in arxiv_topics.keys():
        if not os.path.exists(category):
            os.makedirs(category)

def generate_survey_prompt(topic: str) -> str:
    return f"""
You are an advanced scholarly writing assistant. Please write a comprehensive academic survey on the topic: "{topic}". 
Make the survey extremely detailed, approaching the maximum token limit. Include:

1) A thorough introduction and background.
2) Definitions of key terms and core concepts.
3) Historical context and significant milestones.
4) A detailed discussion of current research, major subtopics, and important developments.
5) Comparisons of competing or complementary approaches.
6) Critical analysis of challenges and open problems.
7) Potential future directions and opportunities for further research.
8) A concise conclusion summarizing the findings.

Use formal academic language, provide relevant references or examples, and ensure the writing is coherent and well-structured. 
Output everything as a single, continuous text.
"""

def main():
    create_directories()
    client = getQwenClient()

    for category, topics in arxiv_topics.items():
        for topic in topics:
            print(f"Generating survey for topic: {topic} (Category: {category})")
            prompt = generate_survey_prompt(topic)
            survey_text = generateResponse(client, prompt)
            filename = f"survey_{topic}.md"
            file_path = os.path.join(category, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(survey_text)

            print(f"Saved survey for '{topic}' to: {file_path}")

if __name__ == "__main__":
    main()
