from embedding import *
from milvusdb import *

import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI

load_dotenv()

FPT_MODEL = os.getenv("FPT_MODEL")
FPT_API_KEY = os.getenv("FPT_API_KEY")
FPT_BASE_URL = os.getenv("FPT_BASE_URL")

embedModel = Jina()
llm = ChatOpenAI(
        model=FPT_MODEL,
        api_key=FPT_API_KEY,
        base_url=FPT_BASE_URL,
        temperature=0.2,
        max_tokens=2048,
    )

def retrieve_context(query: str, top_k: int = 5) -> str:
    """retrieval"""
    collection = get_milvus_collection()
    collection.load()

    query_vector = embedModel.EmbeddingBysentence(query)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    results = collection.search(
        data = [query_vector],
        anns_field = "vector",
        param = search_params,
        limit = top_k,
        output_fields = ["text"]
    )


    retrieved_texts = []
    for hits in results:
        for hit in hits:
            retrieved_texts.append(hit.entity.get("text"))

    return "\n\n".join(retrieved_texts)


def run_rag(query: str):
    print(f"Cau hoi: {query}")
    print("dang retrieve")

    context = retrieve_context(query, top_k=2)
    prompt = f"""Bạn là một trợ lý ảo thông minh. Dựa vào thông tin được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng. Nếu thông tin không có, hãy nói là bạn không biết. Và đừng dùng markdown

[THÔNG TIN CUNG CẤP]:
{context}

[CÂU HỎI]:
{query}

[TRẢ LỜI]:"""

    print("\n[PROMPT SẴN SÀNG GỬI CHO LLM]:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)

    response = llm.invoke(prompt)
    return response.content


# Chạy thử
if __name__ == "__main__":
    result = run_rag("Chi phí hoạt động ngân hàng năm 2025 như thế nào")
    print(result)



    