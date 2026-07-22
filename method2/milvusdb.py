from pymilvus import MilvusClient, DataType
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, DIMENSION

def get_collection():
    # Kết nối Milvus
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    # Nếu collection đã tồn tại thì trả về client
    if client.has_collection(COLLECTION_NAME):
        print(f"[*] Collection {COLLECTION_NAME} đã tồn tại")
        return client

    # Khai báo schema
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=20)   # "text" hoặc "chart"
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIMENSION)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("image_path", DataType.VARCHAR, max_length=500)
    schema.add_field("tabular_data", DataType.VARCHAR, max_length=65535)

    # Tạo collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        description="Unified Multimodal Collection"
    )

    # Tạo index cho field vector
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128}
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )

    print(f"[*] Đã khởi tạo Milvus Collection: {COLLECTION_NAME}")
    return client

if __name__ == "__main__":
    get_collection()
