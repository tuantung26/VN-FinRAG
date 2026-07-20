from milvusdb import get_milvus_collection
from chunking import chunkByWords
from embedding import *

embedModel = Jina()

def process_and_store_document(file_path: str):
    with open(file_path, "r", encoding = "utf-8") as f:
        raw_text = f.read()

    chunks = chunkByWords(raw_text, chunk_size=300, overlap=50)    
    print("embedding...")
    vector = [embedModel.EmbeddingBysentence(chunk) for chunk in chunks]
    collection = get_milvus_collection()
    data = [{"vector": v, "text": c} for v, c in zip(vector, chunks)]
    insert_result = collection.insert(data)
    collection.load()
    print(f"Đã chèn thêm: {insert_result.insert_count} records.")


if __name__ == "__main__":
    test_file = "result.txt"
    process_and_store_document(test_file)
