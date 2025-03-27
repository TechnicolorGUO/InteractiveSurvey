from asg_loader import DocumentLoading
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