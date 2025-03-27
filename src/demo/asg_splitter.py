from .asg_loader import DocumentLoading
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitting:
    def mineru_recursive_splitter(self, file_path, survey_id, mode):
        docs = DocumentLoading().load_pdf(file_path, survey_id, mode)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=30,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([docs])
        return texts

    def pypdf_recursive_splitter(self, file_path, survey_id):
        docs = DocumentLoading().pypdf_loader(file_path, survey_id)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([docs])
        return texts