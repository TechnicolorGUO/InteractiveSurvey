'''
1. (not important) recursive character text splitter seperators can be determined by the user 如果要将retrieve到的文本应用到demo 需要完整句子
2. chunk size of the splitter needs to be further optimized
'''
from .asg_loader import DocumentLoading
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import SpacyTextSplitter, SentenceTransformersTokenTextSplitter

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import re
import spacy
import time


class TextSplitting:

    def mineru_recursive_splitter(self, file_path, survey_id):
        docs = DocumentLoading().load_pdf(file_path, survey_id)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=30,
            length_function=len,
            is_separator_regex=False,
        )
        # print(docs) # output before splitting
        texts = text_splitter.create_documents([docs])
        return texts # output after splitting
        # print(len(texts))
        # for text in texts:
        #     print(text) # visualizing the output
        #     print("==============================")

        # splits = []
        # for doc in docs:
        #     doc_splits = text_splitter.split_text(doc)
        #     splits.extend(doc_splits)
        # return splits

    def txt_character(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            docs = f.read()
        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=300,
            chunk_overlap=0,
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

    def unstructured_recursive_splitter(self, file_path, survey_id):
        doc_start_time = time.time()
        docs = DocumentLoading().unstructured_loader(file_path, survey_id)
        print("Document loading time: ", time.time() - doc_start_time)

        split_start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([docs])
        print("Splitting time: ", time.time() - split_start_time)
        return texts

    # openai api needed
    def pypdf_semantic_splitter(self, file_path, survey_id):
        docs = DocumentLoading().pypdf_loader(file_path, survey_id)
        text_splitter = SemanticChunker(OpenAIEmbeddings())
        texts = text_splitter.create_documents([docs])
        # print(texts[0].page_content)
        for text in texts:
            print(text)
            print("==============================")

    def spacy_token_splitter(self, file_path):
        docs = DocumentLoading().pypdf_loader(file_path)
        text_splitter = SpacyTextSplitter(chunk_size=400)
        texts = text_splitter.split_text(docs)
        # texts = text_splitter.create_documents([docs])
        for text in texts:
            print(text)
            print("==============================")

    def tiktoken_token_splitter(self, file_path):
        docs = DocumentLoading().pypdf_loader(file_path)
        # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        #     encoding="cl100k_base", chunk_size=100, chunk_overlap=0
        # )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4", # 使用 GPT-4 的标记化规则
            chunk_size=300,
            chunk_overlap=20,
        )
        # texts = text_splitter.create_documents([docs])
        texts = text_splitter.split_text(docs)
        for text in texts:
            print(text)
            print("==============================")
        # print(texts[0])

    def sentencetransformers_token_splitter(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            docs = f.read()
        text_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0,
            model_name="sentence-transformers/all-mpnet-base-v2",
            tokens_per_chunk=512
        )
        texts = text_splitter.create_documents([docs])
        for text in texts:
            print(text)
            print("==============================")

if __name__ == "__main__":
    asg_splitter = TextSplitting()

    asg_splitter.unstructured_recursive_splitter("./Test1.pdf")
    # asg_splitter.unstructured_recursive_splitter("./Test2.pdf")
    # asg_splitter.unstructured_recursive_splitter("./Test3.pdf")
    # asg_splitter.unstructured_recursive_splitter("./Test5.pdf")
    # asg_splitter.unstructured_recursive_splitter("./Test6.pdf")
    # asg_splitter.unstructured_recursive_splitter("./Test7.pdf")
    # asg_splitter.unstructured_recursive_splitter("./Test8.pdf")