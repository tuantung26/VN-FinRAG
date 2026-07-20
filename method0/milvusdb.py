from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "vectorDatabase"
DIMENSION = 1024

def get_milvus_collection() -> Collection:
    """Ket noi va tra ve collection, neu chua co thi tao"""
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        return Collection(name=COLLECTION_NAME)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="vector Database")
    collection = Collection(name=COLLECTION_NAME, schema=schema)


    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    collection.create_index("vector", index_params)
    print("The collection is successfully created")

    return collection

    
    
    
    