# 0: no papers found
# 1: very few papers found (<20)
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


import os
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time
import re
import urllib.parse
import requests
from asg_query import generate_query_qwen
from dotenv import load_dotenv

load_dotenv()

PARENT_FOLDER = "arxiv_downloads"
os.makedirs(PARENT_FOLDER, exist_ok=True)

def sanitize_filename(filename):
    """去除非法字符 确保文件名可用"""
    filename = filename.replace("\n", "").strip()  # 去掉换行符
    filename = re.sub(r'[\/:*?"<>|]', '_', filename)  # 统一替换特殊字符
    return filename[:100] + ".pdf"  # 限制文件名长度 避免过长

def search_arxiv_papers(topic, max_results=50):
    """查询 arXiv API 获取某个 topic 相关的论文"""
    query_qwen = generate_query_qwen(topic)
    encoded_query = urllib.parse.quote_plus(query_qwen)  # URL 编码
    url = f"https://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate"

    # base_url = "http://export.arxiv.org/api/query?"
    # query = f"search_query=all:{topic.replace(' ', '+')}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    
    # url = base_url + query
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching data for {topic}: {response.status_code}")
        return []
    
    root = ET.fromstring(response.text)
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    
    papers = []
    for entry in entries:
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        pdf_link = entry.find("{http://www.w3.org/2005/Atom}id").text.replace("abs", "pdf")
        papers.append({"title": title, "pdf_link": pdf_link})
    
    return papers

def download_pdf(url, folder, filename):
    """下载 PDF 并保存到指定文件夹"""
    file_path = os.path.join(folder, filename)
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print(f"Failed to download {url}")

def download_arxiv_papers(topic, max_results=50):
    """下载指定 topic 的 arXiv 论文"""
    # folder_name = topic.replace(" ", "_")
    folder_name = os.path.join(PARENT_FOLDER, topic.replace(" ", "_"))
    os.makedirs(folder_name, exist_ok=True)

    papers = search_arxiv_papers(topic, max_results)

    if not papers:
        print(f"No papers found for topic: {topic}")
        return
    
    print(f"Downloading {len(papers)} papers for topic: {topic}")

    for paper in tqdm(papers, total=len(papers)):
        filename = sanitize_filename(paper['title'])
        pdf_link = paper["pdf_link"]
        download_pdf(pdf_link, folder_name, filename)
        time.sleep(2)  # 避免请求过快被限制
        
    print(f"Download complete. Papers saved in: {folder_name}")

all_topics = []
for primary, subtopics in arxiv_topics.items():
    all_topics.extend(subtopics)

# 试试
for topic in all_topics:
    print(f"\nProcessing topic: {topic}")
    download_arxiv_papers(topic, max_results=50)