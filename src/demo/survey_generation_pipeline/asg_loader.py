# import spacy
# from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader

# MinerU (Shanghai AI Lab) 
import os
import re
import json
import subprocess
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
import shutil

# load spaCy model
# nlp = spacy.load("en_core_web_sm")

class DocumentLoading:
    def convert_pdf_to_md(self, pdf_file, output_dir="output", method="auto"):
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        target_dir = os.path.join(output_dir, base_name)
        md_file_path = os.path.join(target_dir, method, f"{base_name}.md")  # Markdown 文件路径
        print("The md file path is: ", md_file_path)

        # 检查是否已存在 Markdown 文件
        if os.path.exists(md_file_path):
            print(f"Markdown file for {pdf_file} already exists at {md_file_path}. Skipping conversion.", flush=True)
            return
            
        # 执行转换命令
        command = ["mineru", "-p", pdf_file, "-o", output_dir, "-m", method]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.communicate(timeout=300)  # 设置超时时间为 60 秒
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
            process.terminate()  # 终止子进程
            process.wait()  # 等待子进程退出
            # 清理资源
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
        except Exception as e:
            print(f"An error occurred: {e}")
            if process.poll() is None:  # 如果子进程仍在运行
                process.terminate()
                process.wait()
            # 清理资源
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
    # new
    def convert_pdf_to_md_new(self, pdf_dir, output_dir="output", method="auto"):
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

        for pdf_file in pdf_files:
            base_name = os.path.splitext(os.path.basename(pdf_file))[0]
            target_dir = os.path.join(output_dir, base_name)

            if os.path.exists(target_dir):
                print(f"Folder for {pdf_file} already exists in {output_dir}. Skipping conversion.")
            else:
                command = ["mineru", "-p", pdf_file, "-o", output_dir, "-m", method]
                try:
                    subprocess.run(command, check=True)
                    print(f"Successfully converted {pdf_file} to markdown format in {target_dir}.")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred: {e}")

    def batch_convert_pdfs(pdf_files, output_dir="output", method="auto", max_workers=None):
        # Create a process pool to run the conversion in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit each PDF file to the process pool for conversion
            futures = [executor.submit(convert_pdf_to_md, pdf, output_dir, method) for pdf in pdf_files]

            # Optionally, you can monitor the status of each future as they complete
            for future in futures:
                try:
                    future.result()  # This will raise any exceptions that occurred during the processing
                except Exception as exc:
                    print(f"An error occurred during processing: {exc}")

    def extract_information_from_md(self, md_text):
            # Title: 在第一个双换行符之前的内容。
            title_match = re.search(r'^(.*?)(\n\n|\Z)', md_text, re.DOTALL)
            title = title_match.group(1).strip() if title_match else "N/A"

            # Authors: 从第一个双换行符之后，到 Abstract 标志之前的内容。
            authors_match = re.search(
                r'\n\n(.*?)(\n\n[aA][\s]*[bB][\s]*[sS][\s]*[tT][\s]*[rR][\s]*[aA][\s]*[cC][\s]*[tT][^\n]*\n\n)', 
                md_text, 
                re.DOTALL
            )
            authors = authors_match.group(1).strip() if authors_match else "N/A"

            # Abstract: 从 Abstract 标志之后，到下一个双换行符之前的内容。
            abstract_match = re.search(
                r'(\n\n[aA][\s]*[bB][\s]*[sS][\s]*[tT][\s]*[rR][\s]*[aA][\s]*[cC][\s]*[tT][^\n]*\n\n)(.*?)(\n\n|\Z)', 
                md_text, 
                re.DOTALL
            )
            abstract = abstract_match.group(0).strip() if abstract_match else "N/A"
            abstract = re.sub(r'^[aA]\s*[bB]\s*[sS]\s*[tT]\s*[rR]\s*[aA]\s*[cC]\s*[tT][^\w]*', '', abstract)
            abstract = re.sub(r'^[^a-zA-Z]*', '', abstract)

            # Introduction
            introduction_match = re.search(
                r'\n\n([1I][\.\- ]?\s*)?[Ii]\s*[nN]\s*[tT]\s*[rR]\s*[oO]\s*[dD]\s*[uU]\s*[cC]\s*[tT]\s*[iI]\s*[oO]\s*[nN][\.\- ]?\s*\n\n(.*?)'
                r'(?=\n\n(?:([2I][I]|\s*2)[^\n]*?\n\n|\n\n(?:[2I][I][^\n]*?\n\n)))',
                md_text, 
                re.DOTALL
            )
            introduction = introduction_match.group(2).strip() if introduction_match else "N/A"

            # Main Content: 提取全文直到 References 部分为止
            main_content_match = re.search(
                r'(.*?)(\n\n([3I][\.\- ]?\s*)?[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee][Ss][^\n]*\n\n|\Z)', 
                md_text, 
                re.DOTALL
            )
            
            if main_content_match:
                main_content = main_content_match.group(1).strip()
            else:
                main_content = "N/A"

            # 将提取的数据存入字典中
            extracted_data = {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "introduction": introduction,
                "main_content": main_content
            }
            return extracted_data
    
    def process_md_file(self, md_file_path, survey_id, txt_path='./src/static/data/txt/'):
        loader = UnstructuredMarkdownLoader(md_file_path)
        data = loader.load()
        assert len(data) == 1, "Expected exactly one document in the markdown file."
        assert isinstance(data[0], Document), "The loaded data is not of type Document."
        extracted_text = data[0].page_content
        
        extracted_data = self.extract_information_from_md(extracted_text)
        if len(extracted_data["abstract"]) < 10:
            extracted_data["abstract"] = extracted_data['title']

        title = os.path.splitext(os.path.basename(md_file_path))[0]
        title_new = title.strip()
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '_']
        for char in invalid_chars:
            title_new = title_new.replace(char, ' ')
        # print("============================")
        # print(title_new)
        os.makedirs(f'{txt_path}/{survey_id}', exist_ok=True)
        with open(f'{txt_path}/{survey_id}/{title_new}.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        # print(extracted_data)
        return extracted_data['introduction']
    
    def process_md_file_full(self, md_file_path, survey_id, txt_path='./src/static/data/txt/'):
        loader = UnstructuredMarkdownLoader(md_file_path)
        data = loader.load()
        assert len(data) == 1, "Expected exactly one document in the markdown file."
        assert isinstance(data[0], Document), "The loaded data is not of type Document."
        extracted_text = data[0].page_content
        
        extracted_data = self.extract_information_from_md(extracted_text)
        if len(extracted_data["abstract"]) < 10:
            extracted_data["abstract"] = extracted_data['title']

        title = os.path.splitext(os.path.basename(md_file_path))[0]
        title_new = title.strip()
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '_']
        for char in invalid_chars:
            title_new = title_new.replace(char, ' ')
        # print("============================")
        # print(title_new)
        os.makedirs(f'{txt_path}/{survey_id}', exist_ok=True)
        with open(f'{txt_path}/{survey_id}/{title_new}.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        # print(extracted_data)
        return extracted_data['abstract'] + extracted_data['introduction'] + extracted_data['main_content']

    
    def load_pdf(self, pdf_file, survey_id, mode):
        os.makedirs(f'./src/static/data/md/{survey_id}', exist_ok=True)
        output_dir = f"./src/static/data/md/{survey_id}"
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        target_dir = os.path.join(output_dir, base_name, "auto")

        # 1. Convert PDF to markdown if the folder doesn't exist
        self.convert_pdf_to_md(pdf_file, output_dir)

        # 2. Process the markdown file in the output directory
        md_file_path = os.path.join(target_dir, f"{base_name}.md")
        if not os.path.exists(md_file_path):
            raise FileNotFoundError(f"Markdown file {md_file_path} does not exist. Conversion might have failed.")

        if mode == "intro":
            return self.process_md_file(md_file_path, survey_id)
        elif mode == "full":
            return self.process_md_file_full(md_file_path, survey_id)

    # wrong, still being tested
    def load_pdf_new(self, pdf_dir, survey_id):
        os.makedirs(f'./src/static/data/md/{survey_id}', exist_ok=True)
        output_dir = f"./src/static/data/md/{survey_id}"
        self.convert_pdf_to_md_new(pdf_dir, output_dir)
        markdown_files = glob.glob(os.path.join(output_dir, "*", "auto", "*.md"))
        all_introductions = []
        
        for md_file_path in markdown_files:
            try:
                introduction = self.process_md_file(md_file_path, survey_id)
                all_introductions.append(introduction)
            except FileNotFoundError as e:
                print(f"Markdown file {md_file_path} does not exist. Conversion might have failed.")
        
        return all_introductions



    def parallel_load_pdfs(self, pdf_files, survey_id, max_workers=4):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for parallel execution
            futures = [executor.submit(self.load_pdf, pdf, survey_id) for pdf in pdf_files]
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    print(f"Processed result: {result}")
                except Exception as e:
                    print(f"Error processing PDF: {e}")
    
    def ensure_non_empty_introduction(self, introduction, full_text):
        """
        Ensure introduction is not empty. If empty, replace with full text.
        """
        if introduction == "N/A" or len(introduction.strip()) < 50:
            return full_text.strip()
        return introduction

    def extract_information_from_md_new(self, md_text):
        # Title extraction
        title_match = re.search(r'^(.*?)(\n\n|\Z)', md_text, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "N/A"

        # Authors extraction
        authors_match = re.search(
            r'\n\n(.*?)(\n\n[aA][\s]*[bB][\s]*[sS][\s]*[tT][\s]*[rR][\s]*[aA][\s]*[cC][\s]*[tT][^\n]*\n\n)', 
            md_text, 
            re.DOTALL
        )
        authors = authors_match.group(1).strip() if authors_match else "N/A"

        # Abstract extraction
        abstract_match = re.search(
            r'(\n\n[aA][\s]*[bB][\s]*[sS][\s]*[tT][\s]*[rR][\s]*[aA][\s]*[cC][\s]*[tT][^\n]*\n\n)(.*?)(\n\n|\Z)', 
            md_text, 
            re.DOTALL
        )
        abstract = abstract_match.group(0).strip() if abstract_match else "N/A"
        abstract = re.sub(r'^[aA]\s*[bB]\s*[sS]\s*[tT]\s*[rR]\s*[aA]\s*[cC]\s*[tT][^\w]*', '', abstract)
        abstract = re.sub(r'^[^a-zA-Z]*', '', abstract)

        # Introduction extraction
        introduction_match = re.search(
            r'\n\n([1I][\.\- ]?\s*)?[Ii]\s*[nN]\s*[tT]\s*[rR]\s*[oO]\s*[dD]\s*[uU]\s*[cC]\s*[tT]\s*[iI]\s*[oO]\s*[nN][\.\- ]?\s*\n\n(.*?)',
            md_text, re.DOTALL
        )
        introduction = introduction_match.group(2).strip() if introduction_match else "N/A"

        # Ensure introduction is not empty
        introduction = self.ensure_non_empty_introduction(introduction, md_text)

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "introduction": introduction
        }



    # def extract_names(self, text):
    #     '''
    #     Extract names from text using spaCy
    #     '''
    #     doc = nlp(text)
    #     names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    #     return list(set(names))
    
    # def clean_authors_text(self, text):
    #     '''
    #     Clean authors text by removing emails and academic keywords
    #     '''
    #     # clear emails
    #     cleaned_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    #     # clear academic keywords
    #     academic_keywords = [
    #         r'\bInstitute\b', r'\bUniversity\b', r'\bDepartment\b', r'\bCollege\b', r'\bSchool\b', r'\bFaculty\b', r'\bLaboratory\b',
    #         r'\bResearch\b', r'\bCenter\b', r'\bCentre\b', r'\bProgram\b', r'\bProgramme\b', r'\bGroup\b', r'\bDivision\b',
    #         r'\bUnit\b', r'\bCampus\b', r'\bBuilding\b', r'\bRoom\b', r'\bStreet\b', r'\bAvenue\b', r'\bBoulevard\b', r'\bRoad\b',
    #         r'\bCity\b', r'\bTown\b', r'\bState\b', r'\bProvince\b', r'\bCountry\b', r'\bZip\b', r'\bCode\b', r'\bPostal\b',
    #         r'\bBox\b', r'\bPhone\b', r'\bFax\b', r'\bEmail\b', r'\bWeb\b', r'\bSite\b', r'\bURL\b', r'\bAddress\b', r'\bContact\b',
    #         r'\bFloor\b', r'\bSuite\b', r'\bArea\b', r'\bDistrict\b', r'\bBlock\b'
    #     ]
    #     cleaned_text = re.sub('|'.join(academic_keywords), '', cleaned_text, flags=re.IGNORECASE)
    #     # clear extra spaces
    #     cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    #     return cleaned_text
    
    # def extract_information_unstructured(self, text):
    #     '''
    #     Extract title, authors, abstract, and introduction for unstructured loader
    #     '''
    #     data = {
    #         'title': "NONE",
    #         'authors': "NONE",
    #         'abstract': "NONE",
    #         'introduction': "NONE"
    #     }

    #     title_match = re.search(r'^(.*?)(?:\n|$)', text)
    #     if title_match:
    #         data['title'] = title_match.group(1).strip()
        
    #     authors_match = re.search(r'\n\n(.*?)(?:\n\nABSTRACT|\n\nAbstract|\n\nAbstract\.|\n\nAbstract—)', text, re.DOTALL)
    #     if authors_match:
    #         authors_text = authors_match.group(1).replace('\n', ' ').strip()
    #         cleaned_authors_text = self.clean_authors_text(authors_text)
    #         authors = self.extract_names(cleaned_authors_text)
    #         authors = [re.sub(r'\d+$', '', author).strip() for author in authors]  # clear trailing numbers
    #         data['authors'] = ", ".join(authors)
        
    #     # 适用于search内容刚好匹配到abstract等关键字 而非文本块中包含关键字即可search (Test7.pdf中的abstract部分)无法正确提取
    #     abstract_match = re.search(
    #         r'(?i)ABSTRACT\s*(.*?)(?=\n\s*1\s*\.?\s*I\s*N\s*T\s*R\s*O\s*D\s*U\s*C\s*T\s*I\s*O\s*N|\n\s*INTRODUCTION|\n\s*1\s*INTRODUCTION|\n\s*1\s*I\s*N\s*T\s*R\s*O\s*D\s*U\s*C\s*T\s*I\s*O\s*N)',
    #         text, re.DOTALL
    #     )       
    #     if abstract_match:
    #         abstract_text = abstract_match.group(1).strip()
    #         # clear leading dot
    #         if abstract_text.startswith('.'):
    #             abstract_text = abstract_text[1:].strip()
    #         data['abstract'] = abstract_text
        
    #     intro_match = re.search(
    #         r'(?i)(?:1\s*\.?\s*INTRODUCTION|1\s*\.?\s*I\s*N\s*T\s*R\s*O\s*D\s*U\s*C\s*T\s*I\s*O\s*N|INTRODUCTION|I\s*N\s*T\s*R\s*O\s*D\s*U\s*C\s*T\s*I\s*O\s*N)\s*(.*?)(?=\n\s*2\s*\.?\s*R\s*E\s*L\s*A\s*T\s*E\s*D\s*W\s*O\s*R\s*K|\n\s*2\s*\.?\s*RELATED\s*WORK|\n\s*RELATED\s*WORK|\n\s*R\s*E\s*L\s*A\s*T\s*E\s*D\s*W\s*O\s*R\s*K)',
    #         text, re.DOTALL
    #     )
    #     if intro_match:
    #         data['introduction'] = intro_match.group(1).strip()
    #     return data

    # def extract_information_pypdf(self, text):
    #     '''
    #     Extract title, authors, abstract, and introduction for pypdf loader
    #     '''
    #     data = {
    #         'title': "NONE",
    #         'authors': "NONE",
    #         'abstract': "NONE",
    #         'introduction': "NONE"
    #     }

    #     # assume title is the first 2 lines
    #     title_match = re.search(r'^(.*?)\n(.*?)\n', text, re.DOTALL)
    #     if title_match:
    #         data['title'] = title_match.group(1).strip() + ' ' + title_match.group(2).strip()

    #     authors_match = re.search(r'\n(.+?)\n(?:\n|ABSTRACT|INTRODUCTION)', text, re.DOTALL)

    #     if authors_match:
    #         data['authors'] = authors_match.group(1).replace('\n', ' ').strip()
    #         print(data['authors'])
    #     else:
    #         print("Authors not found")

    #     abstract_match = re.search(
    #         r'(?i)ABSTRACT\s*(.*?)(?=\n\s*\d+\.\s*INTRODUCTION|\n\s*\d+\.\s*Introduction|\n\s*\d+\sINTRODUCTION|\n\s*\d+\sIntroduction)',
    #         text, re.DOTALL
    #     )
    #     if abstract_match:
    #         abstract_text = abstract_match.group(1).strip()
    #         if abstract_text.startswith('.'):
    #             abstract_text = abstract_text[1:].strip()
    #         data['abstract'] = abstract_text

    #     intro_match = re.search(r'(?i)(?:\d+\.\s*INTRODUCTION|\d+\s*INTRODUCTION|INTRODUCTION)\s*(.*?)(?=\n\s*\d+\.\s*\w+|\n\s*\w+\n|$)', text, re.DOTALL)
    #     if intro_match:
    #         data['introduction'] = intro_match.group(1).strip()
    #     return data

    # def clean_headers_footers(self, text_blocks):
    #     '''
    #     Clean headers and footers from text blocks
    #     '''
    #     block_count = {}
    #     for block in text_blocks:
    #         if block in block_count:
    #             block_count[block] += 1
    #         else:
    #             block_count[block] = 1

    #     # clear blocks that appear in every page (likely headers/footers)
    #     header_footer_blocks = {block for block, count in block_count.items() if count > 1}
    #     cleaned_blocks = [block for block in text_blocks if block not in header_footer_blocks]
    #     return cleaned_blocks

    # # 处理页边 页面注释等 枚举
    # def clean_annotations(self, text_blocks):
    #     '''
    #     Clean annotations from text blocks
    #     '''
    #     cleaned_blocks = []
    #     for block in text_blocks:
    #         # clear blocks that are likely annotations
    #         if re.search(r'\d{4}\s\d+\s\w+\sConference\s.*?\|\s.*?\|\sDOI:.*?\s\|\s\w+:\s.*?\n', block, flags=re.DOTALL) or \
    #         re.search(r'http\S+', block) or \
    #         re.search(r'\d+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+', block, flags=re.DOTALL):
    #             continue
    #         cleaned_blocks.append(block)
    #     return cleaned_blocks

    # def clean_hyphenation_unstructured(self, text):
    #     '''
    #     Clean hyphenation from text for unstructured loader
    #     '''
    #     # remove hyphenation across lines
    #     text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    #     # remove hyphenation within lines
    #     text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)        
    #     return text
    
    # def clean_hyphenation_pypdf(self, text):
    #     '''
    #     Clean hyphenation from text for pypdf loader
    #     '''
    #     # remove hyphenation across lines, ensuring no space between words
    #     text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    #     return text

    # def clean_empty_and_numeric_lines(self, text):
    #     '''
    #     Clean empty and numeric lines from text
    #     '''
    #     lines = text.split('\n')
    #     cleaned_lines = []
    #     for line in lines:
    #         if not line.strip() or line.strip().isdigit():
    #             continue
    #         cleaned_lines.append(line)
    #     return "\n".join(cleaned_lines)
    
    # def remove_newlines(self, text):
    #     '''
    #     Remove newlines from text
    #     '''
    #     return text.replace('\n', ' ')

    # # extract using pypdfloader
    # def pypdf_loader_extract(self, file_path):
    #     '''
    #     Single '\\n' between sections \n
    #     Extract title, authors, abstract, introduction using pypdf loader
    #     '''
    #     loader = PyPDFLoader(file_path)
    #     documents = loader.load()

    #     text_blocks = [doc.page_content for doc in documents]
    #     split_blocks = [block.split('\n') for block in text_blocks]
    #     flat_blocks = [item for sublist in split_blocks for item in sublist]
    #     cleaned_blocks = self.clean_headers_footers(flat_blocks)
    #     # cleaned_blocks = self.clean_annotations(cleaned_blocks)
    #     text = "\n".join(cleaned_blocks)
    #     # combine all pages into one text
    #     text = "\n".join([doc.page_content for doc in documents])
    #     # clean empty lines and numeric lines
    #     text = self.clean_empty_and_numeric_lines(text)
    #     # print(text)
    #     extracted_data = self.extract_information_pypdf(text)
    #     extracted_data = {k: self.clean_hyphenation_pypdf(v) for k, v in extracted_data.items()}
    #     extracted_data = {k: self.remove_newlines(v) for k, v in extracted_data.items()}
        
    #     # for extract information in json file
    #     with open('extracted_info.json', 'w', encoding='utf-8') as f:
    #         json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    #     print(extracted_data)    
  
    # def unstructured_loader(self, file_path, survey_id):
    #     '''
    #     Comibine extracted 4 parts into one text (unstructured loader version)
    #     '''
    #     doc_load_time = time.time()
    #     loader = UnstructuredPDFLoader(file_path)
    #     documents = loader.load()

    #     print("Document loading time: ", time.time() - doc_load_time)
    #     # print(documents[0]) # for testing
    #     text_blocks = [doc.page_content for doc in documents]

    #     # Split each page into blocks using double newlines as the delimiter
    #     split_blocks = [block.split('\n\n') for block in text_blocks]
    #     flat_blocks = [item for sublist in split_blocks for item in sublist]
        
    #     cleaned_blocks = self.clean_headers_footers(flat_blocks)
    #     cleaned_blocks = self.clean_annotations(cleaned_blocks)
    #     text = "\n\n".join(cleaned_blocks)
    #     text = self.clean_hyphenation_unstructured(text)
    #     extracted_data = self.extract_information_unstructured(text)
        
    #     # to display 4 sections
    #     extracted_text = extracted_data

    #     title = os.path.splitext(os.path.basename(file_path))[0]

    #     title_new = title.strip()
    #     invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '_']
    #     for char in invalid_chars:
    #         title_new = title_new.replace(char, ' ')
    #     # print("============================")
    #     # print(title_new)
    #     os.makedirs(f'./src/static/data/txt/{survey_id}', exist_ok=True)
    #     with open(f'./src/static/data/txt/{survey_id}/{title_new}.json', 'w', encoding='utf-8') as f:
    #         json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    #     print(extracted_data)

    #     return extracted_text

    # def pypdf_loader(self, file_path, survey_id):
    #     '''
    #     Extract abstract and introduction using pypdf loader \n
    #     Better extraction of abstract and introduction compared with pypdf_loader_old
    #     '''
    #     loader = PyPDFLoader(file_path)
    #     documents = loader.load()
    #     start_extracting = False
    #     extracted_text = ""
    #     temp_text = ""            

    #     for doc in documents:
    #         content = doc.page_content
    #         if not start_extracting:
    #             abstract_match = re.search(r'\bABSTRACT\b', content, re.IGNORECASE)
    #             if abstract_match:
    #                 start_extracting = True
    #                 temp_text += content[abstract_match.start():]  # start from the position of "Abstract"
    #         elif start_extracting:
    #             related_work_match = re.search(r'\bRELATED\s*WORK\b', content, re.IGNORECASE)
    #             if related_work_match:
    #                 temp_text += content[:related_work_match.start()]  # 有时可能在introduction部分中有related work字眼 导致提前结束
    #                 break
    #             else:
    #                 related_work_match_space = re.search(r'\bR\s*E\s*L\s*A\s*T\s*E\s*D\s*W\s*O\s*R\s*K\b', content, re.IGNORECASE)
    #                 if related_work_match_space:
    #                     temp_text += content[:related_work_match_space.start()]  # some spaces not expected, like RELATED to R ELATED
    #                     break
    #             temp_text += content
            
    #     extracted_text = temp_text
    #     extracted_text = self.clean_hyphenation_pypdf(extracted_text)
    #     extracted_text = self.clean_empty_and_numeric_lines(extracted_text)
    #     return extracted_text

# if __name__ == "__main__":
#     asg_loader = DocumentLoading()
#     extracted_text = asg_loader.unstructured_loader("./Test3.pdf")
#     print(extracted_text)