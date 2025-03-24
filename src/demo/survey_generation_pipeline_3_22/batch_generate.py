from main import ASG_system
import os

import os

root_path = "."
cluster_standard = "research method"
pdf_dir = "/home/guest01/develope/web/Auto_Survey_Generator_pdf/arxiv_downloads_new_new_new"
exclude_dir = "/home/guest01/develope/web/Auto_Survey_Generator_pdf/src/demo/survey_generation_pipeline_3_22/result_3_22"

# 获取需要排除的文件夹名称（result 目录下的文件夹，假设它们不含 `_`）
if not os.path.exists(exclude_dir):
    os.makedirs(exclude_dir)
exclude_folders = set(
    folder for folder in os.listdir(exclude_dir) 
    if os.path.isdir(os.path.join(exclude_dir, folder))
)

# 过滤 pdf_dir 目录下的文件夹
survey_titles = [
    folder.replace("_", " ")  # 去除 underscore
    for folder in os.listdir(pdf_dir) 
    if os.path.isdir(os.path.join(pdf_dir, folder)) and folder.replace("_", " ") not in exclude_folders
]

pdf_paths = [
    os.path.join(pdf_dir, folder) 
    for folder in os.listdir(pdf_dir) 
    if os.path.isdir(os.path.join(pdf_dir, folder)) and folder.replace("_", " ") not in exclude_folders
]

# 打印结果以检查
print("Survey Titles:", len(survey_titles))
print("PDF Paths:", len(pdf_paths))

for i in range(len(pdf_paths)):
    asg_system = ASG_system(root_path, survey_titles[i], pdf_paths[i], survey_titles[i], cluster_standard)
    asg_system.parsing_pdfs()
    asg_system.description_generation()
    asg_system.agglomerative_clustering()
    asg_system.outline_generation()
    asg_system.section_generation()
    asg_system.citation_generation()



# if __name__ == "__main__":
#     root_path = "."
#     pdf_path = "./pdfs/test"
#     survey_title = "Automating Literature Review Generation with LLM"
#     cluster_standard = "method"
#     asg_system = ASG_system(root_path, 'test', pdf_path, survey_title, cluster_standard)
#     asg_system.parsing_pdfs()
#     asg_system.description_generation()
#     asg_system.agglomerative_clustering()
#     asg_system.outline_generation()
#     asg_system.section_generation()
#     asg_system.citation_generation()