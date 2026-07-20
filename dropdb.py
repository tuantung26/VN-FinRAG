from pymilvus import MilvusClient
import os
from dotenv import load_dotenv
load_dotenv()

MILVUS_HOST=os.getenv("MILVUS_HOST")
MILVUS_PORT=os.getenv("MILVUS_PORT")
COLLECTION_NAME=os.getenv("COLLECTION_NAME")

client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
client.drop_collection(collection_name=COLLECTION_NAME)
print("Dropped!")