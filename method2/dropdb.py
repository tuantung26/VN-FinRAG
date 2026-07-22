from pymilvus import MilvusClient
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
client.drop_collection(collection_name=COLLECTION_NAME)
print("Dropped!")