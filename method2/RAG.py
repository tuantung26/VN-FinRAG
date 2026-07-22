from llm import get_llm_wandb
from huggingface_hub import get_collection
from milvusdb import *
from embedding import *
from dotenv import load_dotenv
import os
import base64
from pprint import pprint
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage
from config import COLLECTION_NAME


load_dotenv()

llm = get_llm_wandb()

embedding_model = Jina()

def retrieve_context(query: str, top_k: int = 3):
    """retrieval"""
    collection = get_collection()
    collection.load_collection(collection_name=COLLECTION_NAME)

    query_vector = embedding_model.EmbeddingBysentence(query)
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    total_result = collection.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        search_params=search_params,
        limit=top_k,
        output_fields=["text", "image_path", "chunk_type", "tabular_data"]
    )

    retrieve_context = []
    image_paths = []
    tabular_data = []

    for hits in total_result:
        for hit in hits:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {})
            chunk_type = entity.get("chunk_type")
            text = entity.get("text", "")
            img_p = entity.get("image_path", "")
            tab_d = entity.get("tabular_data", "")

            if chunk_type == "chart":
                if img_p and os.path.exists(img_p):
                    image_paths.append(img_p)
                if text:
                    retrieve_context.append(f"[Mô tả biểu đồ]: {text}")
                if tab_d and not tab_d.startswith("loi o cai deplot"):
                    tabular_data.append(tab_d)
            else:
                if chunk_type == "text":
                    retrieve_context.append(text)

    return retrieve_context, image_paths, tabular_data


def run_rag(query: str):
    print(f"Câu hỏi: {query}")
    print("Đang thực hiện retrieval...")

    retrieval = retrieve_context(query, top_k=3)

    context_list = retrieval[0]
    image_paths = retrieval[1]
    tabular_data_list = retrieval[2]

    context_str = "\n\n".join(context_list) if context_list else "Không tìm thấy văn bản phù hợp."
    tabular_str = "\n\n".join(tabular_data_list) if tabular_data_list else "Không có dữ liệu bảng."

    prompt = f"""Bạn là một trợ lý ảo thông minh. Dựa vào thông tin được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng. Nếu thông tin không có, hãy nói là bạn không biết. Và đừng dùng markdown.

[THÔNG TIN CUNG CẤP]:
{context_str}

[BẢNG DỮ LIỆU CUNG CẤP]:
{tabular_str}

[CÂU HỎI]:
{query}

[TRẢ LỜI]:"""

    content_list = [{"type": "text", "text": prompt}]

    for path in image_paths:
        if os.path.exists(path):
            with open(path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                )

    message = [HumanMessage(content=content_list)]

    response = llm.invoke(message)
    return response.content


if __name__ == "__main__":
    answer = run_rag("Doanh thu của xăng không chì RON 95 chênh lệch như thế nào ở năm 2025 và 2024?")
    print("\n--- KẾT QUẢ RAG ---")
    print(answer)