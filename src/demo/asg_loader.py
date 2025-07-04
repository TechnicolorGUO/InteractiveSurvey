import os
import re
import json
import subprocess
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
import shutil

class DocumentLoading:
    def convert_pdf_to_md(self, pdf_file, output_dir="output", method="auto"):
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        target_dir = os.path.join(output_dir, base_name)
        md_file_path = os.path.join(target_dir, method, f"{base_name}.md")
        print("The md file path is: ", md_file_path)

        if os.path.exists(md_file_path):
            print(f"Markdown file for {pdf_file} already exists at {md_file_path}. Skipping conversion.", flush=True)
            return
            
        command = ["mineru", "-p", pdf_file, "-o", output_dir, "-m", method]
        try:
            subprocess.run(command, check=True)
            # 检查是否生成了 Markdown 文件
            if not os.path.exists(md_file_path):
                print(f"Conversion failed: Markdown file not found at {md_file_path}. Cleaning up folder...")
                shutil.rmtree(target_dir)  # 删除生成的文件夹
            else:
                print(f"Successfully converted {pdf_file} to markdown format in {target_dir}.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during conversion: {e}")
            # 如果发生错误且文件夹已生成，则删除文件夹
            if os.path.exists(target_dir):
                print(f"Cleaning up incomplete folder: {target_dir}")
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
        title_match = re.search(r'^(.*?)(\n\n|\Z)', md_text, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "N/A"

        authors_match = re.search(
            r'\n\n(.*?)(\n\n[aA][\s]*[bB][\s]*[sS][\s]*[tT][\s]*[rR][\s]*[aA][\s]*[cC][\s]*[tT][^\n]*\n\n)', 
            md_text, 
            re.DOTALL
        )
        authors = authors_match.group(1).strip() if authors_match else "N/A"

        abstract_match = re.search(
            r'(\n\n[aA][\s]*[bB][\s]*[sS][\s]*[tT][\s]*[rR][\s]*[aA][\s]*[cC][\s]*[tT][^\n]*\n\n)(.*?)(\n\n|\Z)', 
            md_text, 
            re.DOTALL
        )
        abstract = abstract_match.group(0).strip() if abstract_match else "N/A"
        abstract = re.sub(r'^[aA]\s*[bB]\s*[sS]\s*[tT]\s*[rR]\s*[aA]\s*[cC]\s*[tT][^\w]*', '', abstract)
        abstract = re.sub(r'^[^a-zA-Z]*', '', abstract)

        introduction_match = re.search(
            r'\n\n([1I][\.\- ]?\s*)?[Ii]\s*[nN]\s*[tT]\s*[rR]\s*[oO]\s*[dD]\s*[uU]\s*[cC]\s*[tT]\s*[iI]\s*[oO]\s*[nN][\.\- ]?\s*\n\n(.*?)'
            r'(?=\n\n(?:([2I][I]|\s*2)[^\n]*?\n\n|\n\n(?:[2I][I][^\n]*?\n\n)))',
            md_text, 
            re.DOTALL
        )
        introduction = introduction_match.group(2).strip() if introduction_match else "N/A"

        main_content_match = re.search(
            r'(.*?)(\n\n([3I][\.\- ]?\s*)?[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee][Ss][^\n]*\n\n|\Z)', 
            md_text, 
            re.DOTALL
        )
        
        if main_content_match:
            main_content = main_content_match.group(1).strip()
        else:
            main_content = "N/A"

        extracted_data = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "introduction": introduction,
            "main_content": main_content
        }
        return extracted_data
    
    def process_md_file(self, md_file_path, survey_id):
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

        os.makedirs(f'./src/static/data/txt/{survey_id}', exist_ok=True)
        with open(f'./src/static/data/txt/{survey_id}/{title_new}.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        return extracted_data['introduction']
    
    def process_md_file_full(self, md_file_path, survey_id):
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
 
        os.makedirs(f'./src/static/data/txt/{survey_id}', exist_ok=True)
        with open(f'./src/static/data/txt/{survey_id}/{title_new}.json', 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
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