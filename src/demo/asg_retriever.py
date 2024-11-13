import torch
import uuid
import re
import os
import json
import chromadb
from langchain_chroma import Chroma
from .asg_splitter import TextSplitting
from langchain_huggingface import HuggingFaceEmbeddings
from .asg_loader import DocumentLoading
import time
import concurrent.futures

class Retriever:
    client = None
    cur_dir = os.getcwd()  # current directory
    chromadb_path = os.path.join(cur_dir, "chromadb")

    def __init__ (self):
        self.client = chromadb.PersistentClient(path=self.chromadb_path)

    def create_collection_chroma(self, collection_name: str):
        """
        The Collection will be created with collection_name, the name must follow the rules:\n
        0. Collection name must be unique, if the name exists then try to get this collection\n
        1. The length of the name must be between 3 and 63 characters.\n
        2. The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.\n
        3. The name must not contain two consecutive dots.\n
        4. The name must not be a valid IP address.\n
        """
        try: 
            self.client.create_collection(name=collection_name)
        except chromadb.db.base.UniqueConstraintError: 
            self.get_collection_chroma(collection_name)
        return collection_name

    def get_collection_chroma (self, collection_name: str):
        collection = self.client.get_collection(name=collection_name)
        return collection

    def add_documents_chroma (self, collection_name: str, embeddings_list: list[list[float]], documents_list: list[dict], metadata_list: list[dict]) :
        """
        Please make sure that embeddings_list and metadata_list are matched with documents_list\n
        Example of one metadata: {"doc_name": "Test2.pdf", "page": "9"}\n
        The id will be created automatically as uuid v4 
        The chunks content and metadata will be logged (appended) into ./logs/<collection_name>.json
        """
        collection = self.get_collection_chroma(collection_name)
        num = len(documents_list)
        ids=[str(uuid.uuid4()) for i in range(num) ]

        collection.add(
            documents= documents_list,
            metadatas= metadata_list,
            embeddings= embeddings_list,
            ids=ids 
        )
        logpath = os.path.join(self.cur_dir, "logs", f"{collection_name}.json")
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        logs = []
        try:  
            with open (logpath, 'r') as chunklog:
                logs = json.load(chunklog)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logs = [] # old_log does not exist or empty
       
        added_log= [{"chunk_id": ids[i], "metadata": metadata_list[i], "page_content": documents_list[i]} \
                       for i in range(num)]
      
        logs.extend(added_log)

        # write back
        with open (logpath, "w") as chunklog:
            json.dump(logs, chunklog, indent=4)
        print(f"Logged document information to '{logpath}'.")
            
    # def query_chroma (self, collection_name: str, query_embeddings: list[list[float]]) -> dict:
    #     # return n closest results (chunks and metadatas) in order
    #     collection = self.get_collection_chroma(collection_name)
    #     result = collection.query(
    #         query_embeddings=query_embeddings,
    #         n_results=5,
    #     )
    #     print(f"Query executed on collection '{collection_name}'.")
    #     return result
    
    def query_chroma(self, collection_name: str, query_embeddings: list[list[float]], n_results: int = 5) -> dict:
        # return n closest results (chunks and metadatas) in order
        collection = self.get_collection_chroma(collection_name)
        result = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
        # print(f"Query executed on collection '{collection_name}'.")
        return result

    def update_chroma (self, collection_name: str, id_list: list[str], embeddings_list: list[list[float]], documents_list: list[str], metadata_list: list[dict]):
        collection = self.get_collection_chroma(collection_name)
        num = len(documents_list)
        collection.update(
            ids=id_list,
            embeddings=embeddings_list,
            metadatas=metadata_list,
            documents=documents_list,
        )
        update_list = [{"chunk_id": id_list[i], "metadata": metadata_list[i], "page_content": documents_list[i]} for i in range(num)]
       
        # update the chunk log 
        logs = []

        logpath = os.path.join(self.cur_dir, "logs", f"{collection_name}.json")
        # logpath = "{:0}/assets/log/{:1}.json".format(self.cur_dir, collection_name)
        try:  
            with open (logpath, 'r') as chunklog:
                logs = json.load(chunklog)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logs = [] # old_log does not exist or empty, then no need to update
        else:
            for i in range(num):
                for log in logs:
                    if (log["chunk_id"] == update_list[i]["chunk_id"]):
                        log["metadata"] = update_list[i]["metadata"]
                        log["page_content"] = update_list[i]["page_content"]
                        break

        # write back
        with open (logpath, "w") as chunklog:
            json.dump(logs, chunklog, indent=4)
        print(f"Updated log file at '{logpath}'.")

    def delete_collection_entries_chroma(self, collection_name: str, id_list: list[str]):
        collection = self.get_collection_chroma(collection_name)
        collection.delete(ids=id_list)
        print(f"Deleted entries with ids: {id_list} from collection '{collection_name}'.")

    def delete_collection_chroma(self, collection_name: str):
        # delete the collection itself and all entries in the collection 
        print(f"The collection {collection_name} will be deleted forever!")    
        self.client.delete_collection(collection_name)
        try:
            logpath = os.path.join(self.cur_dir, "logs", f"{collection_name}.json")
            print(f"Collection {collection_name} has been removed, deleting log file of this collection")
            os.remove(logpath)
        except FileNotFoundError:
            print("The log of this collection did not exist!")

    def list_collections_chroma(self):
        collections = self.client.list_collections()
        print(f"Existing collections: {[col.name for col in collections]}")

# New function to generate a legal collection name from a PDF filename
def legal_pdf(filename: str) -> str:
    pdf_index = filename.lower().rfind('.pdf')
    if pdf_index != -1:
        name_before_pdf = filename[:pdf_index]
    else:
        name_before_pdf = filename
    name_before_pdf = name_before_pdf.strip()
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name_before_pdf)
    name = name.lower()
    while '..' in name:
        name = name.replace('..', '.')
    name = name[:63]
    if len(name) < 3:
        name = name.ljust(3, '0')  # fill with '0' if the length is less than 3
    if not re.match(r'^[a-z0-9]', name):
        name = 'a' + name[1:]
    if not re.match(r'[a-z0-9]$', name):
        name = name[:-1] + 'a'
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    if ip_pattern.match(name):
        name = 'ip_' + name
    return name

def process_pdf(file_path: str, survey_id: str, embedder: HuggingFaceEmbeddings):
    # Load and split the PDF
    # splitters = TextSplitting().mineru_recursive_splitter(file_path)

    split_start_time = time.time()
    splitters = TextSplitting().mineru_recursive_splitter(file_path, survey_id)

    documents_list = [document.page_content for document in splitters]
    for i in range(len(documents_list)):
        documents_list[i] = documents_list[i].replace('\n', ' ')
    print(f"Splitting took {time.time() - split_start_time} seconds.")

    # Embed the documents
    # embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embed_start_time = time.time()
    doc_results = embedder.embed_documents(documents_list)
    if isinstance(doc_results, torch.Tensor):
        embeddings_list = doc_results.tolist()
    else:
        embeddings_list = doc_results
    print(f"Embedding took {time.time() - embed_start_time} seconds.")

    # Prepare metadata
    metadata_list = [{"doc_name": os.path.basename(file_path)} for i in range(len(documents_list))]

    title = os.path.splitext(os.path.basename(file_path))[0]
    

    title_new = title.strip()
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*','_']
    for char in invalid_chars:
        title_new = title_new.replace(char, ' ')
    print("============================")
    print(title_new)

    # New logic to create collection_name
    # filename = os.path.basename(file_path)
    collection_name = legal_pdf(title_new)

    retriever = Retriever()
    retriever.list_collections_chroma()
    retriever.create_collection_chroma(collection_name)
    retriever.add_documents_chroma(
        collection_name=collection_name,
        embeddings_list=embeddings_list,
        documents_list=documents_list,
        metadata_list=metadata_list
    )

    return collection_name, embeddings_list, documents_list, metadata_list,title_new

def query_embeddings(collection_name: str, query_list: list):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever()

    final_context = ""

    seen_chunks = set()
    for query_text in query_list:
        query_embeddings = embedder.embed_query(query_text)
        query_result = retriever.query_chroma(collection_name=collection_name, query_embeddings=[query_embeddings], n_results=2)

        query_result_chunks = query_result["documents"][0]
        # query_result_ids = query_result["ids"][0]

        for chunk in query_result_chunks:
            if chunk not in seen_chunks:
                final_context += chunk.strip() + "//\n"
                seen_chunks.add(chunk)
    return final_context

# new, may be in parallel
def query_embeddings_new(collection_name: str, query_list: list):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever()

    final_context = ""

    seen_chunks = set()
    def process_query(query_text):
        query_embeddings = embedder.embed_query(query_text)
        query_result = retriever.query_chroma(
            collection_name=collection_name,
            query_embeddings=[query_embeddings],
            n_results=2
        )
        query_result_chunks = query_result["documents"][0]
        return query_result_chunks

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_query, query_text): query_text for query_text in query_list}
        for future in concurrent.futures.as_completed(futures):
            query_result_chunks = future.result()
            for chunk in query_result_chunks:
                if chunk not in seen_chunks:
                    final_context += chunk.strip() + "//\n"
                    seen_chunks.add(chunk)
    return final_context

if __name__ == "__main__":
    print(legal_pdf("blockchain-enabled_federated_learning_data_protection_aggregata.json"))