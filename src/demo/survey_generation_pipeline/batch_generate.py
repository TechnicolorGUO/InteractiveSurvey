import json
import time
from main import ASG_system
import os
from asg_retriever import Retriever


root_path = "."
cluster_standard = "research method"
pdf_dir = "./src/demo/survey_generation_pipeline/pdfs"
exclude_dir = "./src/demo/survey_generation_pipeline/generation_resultds"
arxiv_topics = { # 40/80 topics available in total
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
        "Homotopy Type Theory in Algebraic Topology", # 1
        "Optimal Transport Methods in Probability Theory", # 1
        "Nonlinear Dynamics in Chaotic Systems",
        "Algebraic Geometry in Cryptographic Algorithms", # 1
        "Stochastic Differential Equations in Financial Mathematics", # 1
        "Geometric Group Theory and Low-Dimensional Topology", # 1
        "Information Geometry in Statistical Inference", # 1
        "Network Theory in Combinatorial Optimization" # 1
    ],
    
    "Physics": [
        "Physical Implementation of Quantum Computing",
        "Advances in Gravitational Wave Detection",
        "Topological Properties of Bose-Einstein Condensates",
        "Supersymmetry Theory in High Energy Physics",
        "AI Methods in Programmable Physics Simulations",
        "Dark Matter Modeling in Cosmology",
        "Quantum Entanglement in Condensed Matter Systems", # 1
        "Neutrino Oscillations in Particle Physics",
        "Gravitational Wave Detection and Analysis", # 1
        "Topological Insulators in Solid State Physics" # 1
    ],
    
    "Statistics": [
        "Uncertainty Estimation in Deep Learning",
        "Statistical Physics in Random Matrix Theory",
        "Dimensionality Reduction in High-Dimensional Data",
        "Bayesian Optimization in Machine Learning",
        "Bayesian Hierarchical Models in Genomics", # 1
        "Causal Inference Methods in Epidemiology",
        "High-Dimensional Data Analysis in Machine Learning",
        "Nonparametric Bayesian Methods for Density Estimation" # 1
        "Gaussian Process Regression in Robotics",
        "Statistical Learning Theory and Generalization Error Analysis"
    ],
    
    "Electrical Engineering and Systems Science": [
        "AI Optimization for 5G and 6G Networks",
        "Optimization of Quantum Sensors",
        "Low Power Design Methods in Electronics",
        "5G and Beyond Wireless Communication Technologies", # 1
        "Smart Grid Optimization and Control", # 1
        "Neuromorphic Engineering for Artificial Intelligence", # 1
        "Photonics Integration in Optical Networks", # 1
        "Quantum Dot Technologies in Display Systems",
        "MEMS Sensors for IoT Applications", # 1
        "Terahertz Imaging Systems for Security Screening" # 1
    ],
    
    "Quantitative Biology": [
        "Network-Based Drug Discovery",
        "Computational Modeling in Neuroscience",
        "CRISPR-Cas9 Gene Editing Techniques", # 1
        "Single-Cell RNA Sequencing in Cancer Research",
        "Computational Neuroscience of Brain Connectivity", # 1
        "Synthetic Biology for Metabolic Engineering", # 1
        "Evolutionary Dynamics of Infectious Diseases", # 1
        "Biomechanics of Cellular Motility", # 0
        "Epigenetic Regulation in Stem Cell Differentiation", # 0
        "Systems Biology Approaches to Drug Discovery" # 1
    ],
    
    "Quantitative Finance": [
        "Reinforcement Learning in Algorithmic Trading",
        "Machine Learning for Credit Scoring",
        "Cryptocurrency Market Price Prediction",
        "Algorithmic Trading Strategies in High-Frequency Markets", # 1
        "Blockchain Technologies for Financial Services",
        "Risk Management in Cryptocurrency Investments", # 0
        "Machine Learning for Credit Risk Assessment", # 1
        "Complex Network Analysis in Financial Risk Management", # 0
        "Quantum Computing for Derivatives Pricing", # 1
        "Causal Inference for Portfolio Optimization" # 0
    ],
    
    "Economics": [
        "Modeling Climate Change Economics",
        "Impact of Artificial Intelligence on Labor Markets", # 1
        "Behavioral Economics in Consumer Decision Making", # 0
        "Environmental Economics of Climate Change Policies", # 0
        "Digital Currencies and Monetary Policy", # 1
        "Economic Implications of Global Supply Chain Disruptions", # 0
        "Health Economics of Pandemic Responses", # 1
        "Game Theory Applications in International Trade", # 0
        "Econometric Analysis of Income Inequality", # 1
        "Network Theory in Economic Systems" # 0
    ]
}
# 获取需要排除的文件夹名称（result 目录下的文件夹，假设它们不含 `_`）
if not os.path.exists(exclude_dir):
    os.makedirs(exclude_dir)


# 获取需要排除的文件夹列表
exclude_folders = set(
    folder for folder in os.listdir(exclude_dir) 
    if os.path.isdir(os.path.join(exclude_dir, folder))
)

# 获取所有 survey 标题
survey_titles = [
    arxiv_topics[topic][i] 
    for topic in arxiv_topics
    for i in range(len(arxiv_topics[topic]))
]

# 生成 PDF 路径，并排除 exclude_folders 中的文件夹
pdf_paths = [
    os.path.join(pdf_dir, survey_title)
    for survey_title in survey_titles
    if survey_title not in exclude_folders  # 排除逻辑
]

survey_titles = survey_titles[len(survey_titles)-len(pdf_paths):]

# 打印结果以检查
print("Survey Titles:", len(survey_titles))
print("PDF Paths:", len(pdf_paths))

retriever = Retriever()
for i in range(len(pdf_paths)):
    runtime_json = {}
    asg_system = ASG_system(root_path, survey_titles[i], pdf_paths[i], survey_titles[i], cluster_standard)
    download_time_start = time.time()
    asg_system.download_pdf()
    download_time_end = time.time()
    runtime_json["Download PDF"] = download_time_end - download_time_start

    parsing_time_start = time.time()
    asg_system.parsing_pdfs()
    parsing_time_end = time.time()
    runtime_json["Parsing PDF"] = parsing_time_end - parsing_time_start

    clustering_time_start = time.time()
    asg_system.description_generation(retriever)
    asg_system.agglomerative_clustering()
    clustering_time_end = time.time()
    runtime_json["Clustering"] = clustering_time_end - clustering_time_start

    outline_time_start = time.time()
    asg_system.outline_generation()
    outline_time_end = time.time()
    runtime_json["Outline"] = outline_time_end - outline_time_start

    section_time_start = time.time()
    asg_system.section_generation()
    asg_system.citation_generation()
    section_time_end = time.time()
    runtime_json["Section"] = section_time_end - section_time_start

    runtime_json["Total"] = download_time_end - download_time_start + parsing_time_end - parsing_time_start + clustering_time_end - clustering_time_start + outline_time_end - outline_time_start + section_time_end - section_time_start

    print(runtime_json)

    with open("./src/demo/survey_generation_pipeline/generation_results/" + survey_titles[i] + "_runtime.json", "w") as f:
        json.dump(runtime_json, f, indent=4)