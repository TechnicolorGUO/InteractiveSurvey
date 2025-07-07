import torch
import uuid
import re
import os
import json
import chromadb
from .asg_splitter import TextSplitting
from sentence_transformers import SentenceTransformer
import time
import concurrent.futures
from pathlib import Path

# 禁用 ChromaDB 遥测以避免错误信息
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Singleton retriever to prevent multiple database connections
class RetrieverSingleton:
    _instance = None
    _retriever = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RetrieverSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_retriever(self):
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever
    
    def close_retriever(self):
        if self._retriever is not None:
            try:
                if hasattr(self._retriever, 'client') and self._retriever.client:
                    # ChromaDB doesn't have explicit close method, but we can reset it
                    self._retriever.client = None
            except:
                pass
            self._retriever = None

class Retriever:
    client = None
    cur_dir = os.getcwd()
    chromadb_path = os.path.join(cur_dir, "chromadb")

    def __init__ (self):
        if self.client is None:
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
            with open (logpath, 'r', encoding="utf-8") as chunklog:
                logs = json.load(chunklog)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            logs = []
       
        added_log= [{"chunk_id": ids[i], "metadata": metadata_list[i], "page_content": documents_list[i]} \
                       for i in range(num)]
      
        logs.extend(added_log)

        # write back
        with open (logpath, "w", encoding="utf-8") as chunklog:
            json.dump(logs, chunklog, indent=4)
        print(f"Logged document information to '{logpath}'.")
    
    def query_chroma(self, collection_name: str, query_embeddings: list[list[float]], n_results: int = 5) -> dict:
        # return n closest results (chunks and metadatas) in order
        collection = self.get_collection_chroma(collection_name)
        result = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )
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
        try:  
            with open (logpath, 'r', encoding="utf-8") as chunklog:
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

        with open (logpath, "w", encoding="utf-8") as chunklog:
            json.dump(logs, chunklog, indent=4)
        print(f"Updated log file at '{logpath}'.")

    def delete_collection_entries_chroma(self, collection_name: str, id_list: list[str]):
        collection = self.get_collection_chroma(collection_name)
        collection.delete(ids=id_list)
        print(f"Deleted entries with ids: {id_list} from collection '{collection_name}'.")

    def delete_collection_chroma(self, collection_name: str):
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

# Generate a legal collection name from a PDF filename
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

def process_pdf(file_path: str, survey_id: str, embedder: SentenceTransformer, mode: str):
    # Load and split the PDF
    split_start_time = time.time()
    splitters = TextSplitting().mineru_recursive_splitter(file_path, survey_id, mode)

    documents_list = [document.page_content for document in splitters]
    for i in range(len(documents_list)):
        documents_list[i] = documents_list[i].replace('\n', ' ')
    print(f"Splitting took {time.time() - split_start_time} seconds.")

    # Embed the documents
    embed_start_time = time.time()
    doc_results = embedder.encode(documents_list)
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
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    retriever = Retriever()

    final_context = ""

    seen_chunks = set()
    for query_text in query_list:
        query_embeddings = embedder.encode([query_text])[0].tolist()
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
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    retriever = Retriever()

    final_context = ""

    seen_chunks = set()
    def process_query(query_text):
        query_embeddings = embedder.encode([query_text])[0].tolist()
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

# wza
def query_embeddings_new_new(collection_name: str, query_list: list):
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    retriever = get_retriever()  # Use singleton instead of creating new instance

    final_context = ""  # Stores concatenated context
    citation_data_list = []  # Stores chunk content and collection name as source
    seen_chunks = set()  # Ensures unique chunks are added

    def process_query(query_text):
        # Embed the query text and retrieve relevant chunks
        try:
            query_embeddings = embedder.encode([query_text])[0].tolist()
            query_result = retriever.query_chroma(
                collection_name=collection_name,
                query_embeddings=[query_embeddings],
                n_results=5  # Fixed number of results
            )
            return query_result
        except Exception as e:
            print(f"Error processing query '{query_text}': {e}")
            return None

    # Limit the number of concurrent threads to prevent resource exhaustion
    max_workers = min(len(query_list), 5)  # Limit to 5 concurrent threads
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {executor.submit(process_query, q): q for q in query_list}
        for future in concurrent.futures.as_completed(future_to_query):
            query_text = future_to_query[future]
            try:
                query_result = future.result()
                if query_result is None:
                    continue
            except Exception as e:
                print(f"Query '{query_text}' failed with exception: {e}")
                continue

            if "documents" not in query_result or "distances" not in query_result:
                continue
            if not query_result["documents"] or not query_result["distances"]:
                continue
            docs_list = query_result["documents"][0] if query_result["documents"] else []
            dist_list = query_result["distances"][0] if query_result["distances"] else []

            if len(docs_list) != len(dist_list):
                continue

            for chunk, distance in zip(docs_list, dist_list):
                processed_chunk = chunk.strip()
                if processed_chunk not in seen_chunks:
                    final_context += processed_chunk + "//\n"
                    seen_chunks.add(processed_chunk)
                    citation_data_list.append({
                        "source": collection_name,
                        "distance": distance,
                        "content": processed_chunk,
                    })

    return final_context, citation_data_list

# concurrent version for both collection names and queries
def query_multiple_collections(collection_names: list[str], query_list: list[str], survey_id: str) -> dict:
    """
    Query multiple collections in parallel and return the combined results.

    Args:
        collection_names (list[str]): List of collection names to query.
        query_list (list[str]): List of queries to execute on each collection.

    Returns:
        dict: Combined results from all collections, grouped by collection.
    """
    # Define embedder inside the function
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    retriever = Retriever()

    def query_single_collection(collection_name: str):
        """
        Query a single collection for all queries in the query_list.
        """
        final_context = ""
        seen_chunks = set()

        def process_query(query_text):
            # Embed the query
            query_embeddings = embedder.encode([query_text])[0].tolist()
            # Query the collection
            query_result = retriever.query_chroma(
                collection_name=collection_name,
                query_embeddings=[query_embeddings],
                n_results=5
            )
            query_result_chunks = query_result["documents"][0]
            return query_result_chunks

        # Process all queries in parallel for the given collection
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_query, query_text): query_text for query_text in query_list}
            for future in concurrent.futures.as_completed(futures):
                query_result_chunks = future.result()
                for chunk in query_result_chunks:
                    if chunk not in seen_chunks:
                        final_context += chunk.strip() + "//\n"
                        seen_chunks.add(chunk)

        return final_context

    # Outer parallelism for multiple collections
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(query_single_collection, collection_name): collection_name for collection_name in collection_names}
        for future in concurrent.futures.as_completed(futures):
            collection_name = futures[future]
            results[collection_name] = future.result()

    # Automatically save the results to a JSON file
    import json
    file_path = f'./src/static/data/info/{survey_id}/retrieved_context.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {file_path}")

    return results

# Use singleton retriever in the functions
def get_retriever():
    singleton = RetrieverSingleton()
    return singleton.get_retriever()

def cleanup_retriever():
    """Call this function to cleanup retriever when shutting down"""
    singleton = RetrieverSingleton()
    singleton.close_retriever()