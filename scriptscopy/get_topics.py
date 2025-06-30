import argparse
from datetime import datetime, timedelta
import json
import os
import shutil
import time
import requests
from tqdm import tqdm
from utils import download_arxiv_pdf, getClient, generateResponse, robust_json_parse
from prompts import CATEGORIZE_SURVEY_TITLES, CATEGORIZE_SURVEY_TITLES_SINGLE, EXPAND_CATEGORY_TO_TOPICS, CATEGORIZE_SURVEY_TITLES_HEURISTIC
import arxiv

COARSE_CATEGORIES = [
    "Computer Science",
    "Economics",
    "Electrical Engineering and Systems Science",
    "Mathematics",
    "Physics",
    "Quantitative Biology",
    "Quantitative Finance",
    "Statistics"
]

FINE_CATEGORIES = [
    # Computer Science
    "Artificial Intelligence (cs.AI)",
    "Hardware Architecture (cs.AR)",
    "Computational Complexity (cs.CC)",
    "Computational Engineering, Finance, and Science (cs.CE)",
    "Computational Geometry (cs.CG)",
    "Computation and Language (cs.CL)",
    "Cryptography and Security (cs.CR)",
    "Computer Vision and Pattern Recognition (cs.CV)",
    "Computers and Society (cs.CY)",
    "Databases (cs.DB)",
    "Distributed, Parallel, and Cluster Computing (cs.DC)",
    "Digital Libraries (cs.DL)",
    "Discrete Mathematics (cs.DM)",
    "Data Structures and Algorithms (cs.DS)",
    "Emerging Technologies (cs.ET)",
    "Formal Languages and Automata Theory (cs.FL)",
    "General Literature (cs.GL)",
    "Graphics (cs.GR)",
    "Computer Science and Game Theory (cs.GT)",
    "Human-Computer Interaction (cs.HC)",
    "Information Retrieval (cs.IR)",
    "Information Theory (cs.IT)",
    "Machine Learning (cs.LG)",
    "Logic in Computer Science (cs.LO)",
    "Multiagent Systems (cs.MA)",
    "Multimedia (cs.MM)",
    "Mathematical Software (cs.MS)",
    "Numerical Analysis (cs.NA)",
    "Neural and Evolutionary Computing (cs.NE)",
    "Networking and Internet Architecture (cs.NI)",
    "Other Computer Science (cs.OH)",
    "Operating Systems (cs.OS)",
    "Performance (cs.PF)",
    "Programming Languages (cs.PL)",
    "Robotics (cs.RO)",
    "Symbolic Computation (cs.SC)",
    "Sound (cs.SD)",
    "Software Engineering (cs.SE)",
    "Social and Information Networks (cs.SI)",
    "Systems and Control (cs.SY)",

    # Economics
    "Econometrics (econ.EM)",
    "General Economics (econ.GN)",
    "Theoretical Economics (econ.TH)",

    # Electrical Engineering and Systems Science
    "Audio and Speech Processing (eess.AS)",
    "Image and Video Processing (eess.IV)",
    "Signal Processing (eess.SP)",
    "Systems and Control (eess.SY)",

    # Mathematics
    "Commutative Algebra (math.AC)",
    "Algebraic Geometry (math.AG)",
    "Analysis of PDEs (math.AP)",
    "Algebraic Topology (math.AT)",
    "Classical Analysis and ODEs (math.CA)",
    "Combinatorics (math.CO)",
    "Category Theory (math.CT)",
    "Complex Variables (math.CV)",
    "Differential Geometry (math.DG)",
    "Dynamical Systems (math.DS)",
    "Functional Analysis (math.FA)",
    "General Mathematics (math.GM)",
    "General Topology (math.GN)",
    "Group Theory (math.GR)",
    "Geometric Topology (math.GT)",
    "History and Overview (math.HO)",
    "Information Theory (math.IT)",
    "K-Theory and Homology (math.KT)",
    "Logic (math.LO)",
    "Metric Geometry (math.MG)",
    "Mathematical Physics (math.MP)",
    "Numerical Analysis (math.NA)",
    "Number Theory (math.NT)",
    "Operator Algebras (math.OA)",
    "Optimization and Control (math.OC)",
    "Probability (math.PR)",
    "Quantum Algebra (math.QA)",
    "Rings and Algebras (math.RA)",
    "Representation Theory (math.RT)",
    "Symplectic Geometry (math.SG)",
    "Spectral Theory (math.SP)",
    "Statistics Theory (math.ST)",

    # Physics
    "Cosmology and Nongalactic Astrophysics (astro-ph.CO)",
    "Earth and Planetary Astrophysics (astro-ph.EP)",
    "Astrophysics of Galaxies (astro-ph.GA)",
    "High Energy Astrophysical Phenomena (astro-ph.HE)",
    "Instrumentation and Methods for Astrophysics (astro-ph.IM)",
    "Solar and Stellar Astrophysics (astro-ph.SR)",

    "Disordered Systems and Neural Networks (cond-mat.dis-nn)",
    "Mesoscale and Nanoscale Physics (cond-mat.mes-hall)",
    "Materials Science (cond-mat.mtrl-sci)",
    "Other Condensed Matter (cond-mat.other)",
    "Quantum Gases (cond-mat.quant-gas)",
    "Soft Condensed Matter (cond-mat.soft)",
    "Statistical Mechanics (cond-mat.stat-mech)",
    "Strongly Correlated Electrons (cond-mat.str-el)",
    "Superconductivity (cond-mat.supr-con)",

    "General Relativity and Quantum Cosmology (gr-qc)",

    "High Energy Physics - Experiment (hep-ex)",
    "High Energy Physics - Lattice (hep-lat)",
    "High Energy Physics - Phenomenology (hep-ph)",
    "High Energy Physics - Theory (hep-th)",

    "Mathematical Physics (math-ph)",

    "Adaptation and Self-Organizing Systems (nlin.AO)",
    "Chaotic Dynamics (nlin.CD)",
    "Cellular Automata and Lattice Gases (nlin.CG)",
    "Pattern Formation and Solitons (nlin.PS)",
    "Exactly Solvable and Integrable Systems (nlin.SI)",

    "Nuclear Experiment (nucl-ex)",
    "Nuclear Theory (nucl-th)",

    "Accelerator Physics (physics.acc-ph)",
    "Atmospheric and Oceanic Physics (physics.ao-ph)",
    "Applied Physics (physics.app-ph)",
    "Atomic and Molecular Clusters (physics.atm-clus)",
    "Atomic Physics (physics.atom-ph)",
    "Biological Physics (physics.bio-ph)",
    "Chemical Physics (physics.chem-ph)",
    "Classical Physics (physics.class-ph)",
    "Computational Physics (physics.comp-ph)",
    "Data Analysis, Statistics and Probability (physics.data-an)",
    "Physics Education (physics.ed-ph)",
    "Fluid Dynamics (physics.flu-dyn)",
    "General Physics (physics.gen-ph)",
    "Geophysics (physics.geo-ph)",
    "History and Philosophy of Physics (physics.hist-ph)",
    "Instrumentation and Detectors (physics.ins-det)",
    "Medical Physics (physics.med-ph)",
    "Optics (physics.optics)",
    "Plasma Physics (physics.plasm-ph)",
    "Popular Physics (physics.pop-ph)",
    "Physics and Society (physics.soc-ph)",
    "Space Physics (physics.space-ph)",

    "Quantum Physics (quant-ph)",

    # Quantitative Biology
    "Biomolecules (q-bio.BM)",
    "Cell Behavior (q-bio.CB)",
    "Genomics (q-bio.GN)",
    "Molecular Networks (q-bio.MN)",
    "Neurons and Cognition (q-bio.NC)",
    "Other Quantitative Biology (q-bio.OT)",
    "Populations and Evolution (q-bio.PE)",
    "Quantitative Methods (q-bio.QM)",
    "Subcellular Processes (q-bio.SC)",
    "Tissues and Organs (q-bio.TO)",

    # Quantitative Finance
    "Computational Finance (q-fin.CP)",
    "Economics (q-fin.EC)",
    "General Finance (q-fin.GN)",
    "Mathematical Finance (q-fin.MF)",
    "Portfolio Management (q-fin.PM)",
    "Pricing of Securities (q-fin.PR)",
    "Risk Management (q-fin.RM)",
    "Statistical Finance (q-fin.ST)",
    "Trading and Market Microstructure (q-fin.TR)",

    # Statistics
    "Applications (stat.AP)",
    "Computation (stat.CO)",
    "Methodology (stat.ME)",
    "Machine Learning (stat.ML)",
    "Other Statistics (stat.OT)",
    "Statistics Theory (stat.TH)",
]

category_map = {
        "cs": [
            "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY",
            "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
            "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
            "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
            "cs.SE", "cs.SI", "cs.SD", "cs.SY"
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

def get_top_survey_papers(cats, num=10):
    """
    支持传入单个cat字符串，或cat列表(List[str])
    """
    import arxiv
    if isinstance(cats, str):
        cats = [cats]
    # 构建联合查询
    cat_query = " OR ".join([f"cat:{c}" for c in cats])
    query = f"({cat_query}) AND (ti:survey OR ti:review)"
    search = arxiv.Search(
        query=query,
        max_results=num,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    papers = []
    for result in search.results():
        arxiv_id = result.entry_id.split('/')[-1].split('v')[0]
        papers.append({
            "title": result.title.strip(),
            "arxiv_id": arxiv_id
        })
    return papers

def get_s2_citation(arxiv_id):
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount"
    for _ in range(3):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                # print(f"Fetched citation count for {arxiv_id}: {resp.json().get('citationCount', 0)}")
                return resp.json().get("citationCount", 0)
            elif resp.status_code == 404:
                # print(f"Paper {arxiv_id} not found on Semantic Scholar.")
                return 0
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(1)
    return 0

def get_top_survey_papers_by_citation(
    cats, num=10, oversample=10,
    months_ago_start=36, months_ago_end=3
):
    """
    只考虑发表在 [months_ago_start, months_ago_end] 之间的论文
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=months_ago_start*30)
    end_date = now - timedelta(days=months_ago_end*30)

    # 1. 抓取足够的论文
    arxiv_papers = get_arxiv_papers_in_time_range(
        cats, start_date, end_date, max_results=num*oversample
    )

    # 2. 查询 citation 并排序
    papers = []
    for result in arxiv_papers:
        arxiv_id = result.entry_id.split('/')[-1].split('v')[0]
        citation = get_s2_citation(arxiv_id)
        papers.append({
            "title": result.title.strip(),
            "arxiv_id": arxiv_id,
            "citationCount": citation
        })
        time.sleep(0.2)
    papers.sort(key=lambda x: x["citationCount"], reverse=True)
    return [{"title": p["title"], "arxiv_id": p["arxiv_id"]} for p in papers[:num]]

def is_true_survey_or_review(title, summary):
    """Heuristically filter for real survey/review papers."""
    title = title.lower()
    summary = summary.lower()
    # Exclude common false positives
    bad_keywords = [
        "code reviewer", "reviewer assignment", "reviewer selection", "peer review", "peer-review",
        "reviewing system", "review process", "reviewer recommendation"
    ]
    if any(bad in title for bad in bad_keywords):
        return False
    # Strong positive phrases
    good_phrases = [
        "a survey of", "an overview of", "a review of", "this survey", "this review",
        "comprehensive review", "comprehensive survey", "this paper surveys", "this paper reviews",
        "literature review", "review and prospect", "survey and taxonomy", "survey and analysis"
    ]
    if any(phrase in title for phrase in good_phrases):
        return True
    if any(phrase in summary for phrase in good_phrases):
        return True
    # Allow "survey" in title but require strictness for "review"
    # if "survey" in title:
    #     return True
    return False

def get_arxiv_papers_in_time_range(cats, start_date, end_date, max_results=10):

    """
    Iterate all categories, fetch survey/review, deduplicate and return within time window,
    with additional heuristic filtering for true survey/review papers.
    """
    client = arxiv.Client()
    seen_ids = set()
    unique_papers = []
    for cat in cats:
        query = f"cat:{cat} AND (ti:survey OR ti:review)"
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        for result in client.results(search):
            arxiv_id = result.entry_id.split('/')[-1].split('v')[0]
            published = result.published.replace(tzinfo=None)
            # Heuristic filter here!
            if (
                arxiv_id not in seen_ids
                and start_date <= published <= end_date
                and is_true_survey_or_review(result.title, result.summary)
            ):
                seen_ids.add(arxiv_id)
                unique_papers.append(result)
    print(f"Found {len(unique_papers)} unique, filtered papers in the date range across all cats.")
    return unique_papers

def copy_dataset_to_surveys(systems):
    src_root = os.path.join("outputs", "dataset")
    dst_root = "surveys"
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)
    # 遍历 surveys/一级/主题名 目录（即三级目录）
    for dirpath, dirnames, filenames in os.walk(dst_root):
        rel_path = os.path.relpath(dirpath, dst_root)
        parts = rel_path.split(os.sep)
        # 只在 surveys/一级/主题名 目录下建 system 子文件夹
        if len(parts) == 2:
            for sys_name in systems:
                sys_dir = os.path.join(dirpath, sys_name)
                os.makedirs(sys_dir, exist_ok=True)

def generate_tasks_json(systems, surveys_root="surveys"):
    tasks = {}
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(surveys_root):
        rel_path = os.path.relpath(dirpath, surveys_root)
        parts = rel_path.split(os.sep)
        if len(parts) == 2:
            leaf_dirs.append(dirpath)

    for system in systems:
        system_tasks = []
        for leaf in leaf_dirs:
            folder_name = os.path.basename(leaf)
            system_path = os.path.join(leaf, system)
            abs_system_path = os.path.abspath(system_path).replace(os.sep, '/')
            system_tasks.append({folder_name: abs_system_path})
        tasks[system] = system_tasks

    tasks_json_path = os.path.join(surveys_root, "tasks.json")
    with open(tasks_json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"Generated {tasks_json_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--granularity', choices=['coarse', 'fine'], default='coarse')
    parser.add_argument('--numofsurvey', type=int, default=50)
    parser.add_argument('--systems', nargs='+', default=[], help='List of system names to create subfolders for')
    args = parser.parse_args()

    dataset_dir = os.path.join("outputs", "dataset")
    # ======== 检查 dataset 是否已存在 ==========
    if os.path.exists(dataset_dir):
        print(f"{dataset_dir} exists, skipping data generation.")
        copy_dataset_to_surveys(args.systems)
        generate_tasks_json(args.systems)
        print("Copied existing dataset to surveys/ and created system subfolders.")
        return

    client = getClient()

    if args.granularity == 'coarse':
        # 遍历 category_map 的 key，每个 key 下所有 cat 一起检索
        coarse_surveys_map = {}
        seen_ids_global = set()
        for key in tqdm(category_map, desc="Processing coarse categories"):
            cats = category_map[key]
            print(f"Fetching surveys for categories: {cats}")
            # all_surveys = get_top_survey_papers(cats, args.numofsurvey)
            all_surveys = get_top_survey_papers_by_citation(cats, num=args.numofsurvey, oversample=5)
            # 去重
            unique_surveys = []
            for paper in all_surveys:
                if paper['arxiv_id'] not in seen_ids_global:
                    unique_surveys.append(paper)
                    seen_ids_global.add(paper['arxiv_id'])

            coarse_surveys_map[key] = unique_surveys

            # 聚类
            survey_str = json.dumps(unique_surveys, ensure_ascii=False, indent=2)
            # prompt = CATEGORIZE_SURVEY_TITLES.format(
            #     survey_titles=survey_str,
            #     num_clusters=args.num_per_cat
            # )
            # prompt = CATEGORIZE_SURVEY_TITLES_HEURISTIC.format(
            #     survey_titles=survey_str,
            # )
            prompt = CATEGORIZE_SURVEY_TITLES_SINGLE.format(
                survey_titles=survey_str,
            )


            for attempt in range(3):
                try:
                    raw_response = generateResponse(client, prompt, max_tokens=2048, temerature=0.3)
                    clusters = robust_json_parse(raw_response)
                    break
                except Exception as e:
                    print(f"\nError for clustering '{key}' (attempt {attempt+1}): {e}")
                    if attempt == 2:
                        print(f"Failed to cluster category: {key}, skipping.")
                        clusters = {}
                    else:
                        time.sleep(1)
            # 保存聚类结果
            out_dir = os.path.join("outputs", "dataset", key)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "clusters.json"), 'w', encoding='utf-8') as f:
                json.dump(clusters, f, indent=2, ensure_ascii=False)
            # 下载PDF
            for topic, papers in clusters.items():
                topic_dir = os.path.join(out_dir, topic.replace('/', '_'))
                pdf_dir = os.path.join(topic_dir, "pdfs")
                os.makedirs(pdf_dir, exist_ok=True)
                for paper in papers:
                    try:
                        download_arxiv_pdf(paper['arxiv_id'], pdf_dir)
                    except Exception as e:
                        print(f"Failed to download {paper['arxiv_id']}: {e}")
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/dataset/topics.json", "w", encoding="utf-8") as f:
            json.dump(coarse_surveys_map, f, indent=2, ensure_ascii=False)
        print("Saved outputs/dataset/topics.json")

    elif args.granularity == 'fine':
        seen_ids_global = set()
        fine_surveys_map = {}
        # 遍历 category_map 的 key，每个 key 下每个 cat 单独处理
        for key in tqdm(category_map, desc="Processing fine categories"):
            fine_surveys_map[key] = {}
            for cat in category_map[key]:
                cat_list = [cat]  # get_top_survey_papers_by_citation接收list
                # all_surveys = get_top_survey_papers(cat_list, args.numofsurvey)
                all_surveys = get_top_survey_papers_by_citation(cat_list, num=args.numofsurvey, oversample=10)
                # 去重
                unique_surveys = []
                for paper in all_surveys:
                    if paper['arxiv_id'] not in seen_ids_global:
                        unique_surveys.append(paper)
                        seen_ids_global.add(paper['arxiv_id'])
                # 聚类
                survey_str = json.dumps(unique_surveys, ensure_ascii=False, indent=2)
                # prompt = CATEGORIZE_SURVEY_TITLES.format(
                #     survey_titles=survey_str,
                #     num_clusters=args.num_per_cat
                # )
                # prompt = CATEGORIZE_SURVEY_TITLES_HEURISTIC.format(
                #     survey_titles=survey_str,
                # )
                prompt = CATEGORIZE_SURVEY_TITLES_SINGLE.format(
                    survey_titles=survey_str,
                )
                for attempt in range(3):
                    try:
                        raw_response = generateResponse(client, prompt, max_tokens=2048, temerature=0.3)
                        clusters = robust_json_parse(raw_response)
                        break
                    except Exception as e:
                        print(f"\nError for clustering '{cat}' (attempt {attempt+1}): {e}")
                        if attempt == 2:
                            print(f"Failed to cluster category: {cat}, skipping.")
                            clusters = {}
                        else:
                            time.sleep(1)
                # 保存聚类结果
                out_dir = os.path.join("outputs", "dataset", key, cat)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "clusters.json"), 'w', encoding='utf-8') as f:
                    json.dump(clusters, f, indent=2, ensure_ascii=False)
                # 保存进fine_surveys_map
                fine_surveys_map[key][cat] = clusters
                # 下载PDF
                for topic, papers in clusters.items():
                    topic_dir = os.path.join(out_dir, topic.replace('/', '_'))
                    pdf_dir = os.path.join(topic_dir, "pdfs")
                    os.makedirs(pdf_dir, exist_ok=True)
                    for paper in papers:
                        try:
                            download_arxiv_pdf(paper['arxiv_id'], pdf_dir)
                        except Exception as e:
                            print(f"Failed to download {paper['arxiv_id']}: {e}")
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/dataset/topics.json", "w", encoding="utf-8") as f:
            json.dump(fine_surveys_map, f, indent=2, ensure_ascii=False)
        print("Saved outputs/dataset/topics.json")
    copy_dataset_to_surveys(args.systems)
    generate_tasks_json(args.systems)
    print("Data generation complete. Copied dataset to surveys/ and created system subfolders.")

if __name__ == "__main__":
    main()

#python scripts/main.py --granularity coarse --numofsurvey 10 --systems InteractiveSurvey AutoSurvey SurveyX SurveyForge LLMxMapReduce vanilla