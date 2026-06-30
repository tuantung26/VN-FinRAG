#!/usr/bin/env python3
"""
FILE 2: ONLINE RETRIEVAL-AUGMENTED GENERATION (METHOD 2)
Nhiệm vụ: Chunking nội dung văn bản siêu ngữ nghĩa -> Tìm kiếm tương đồng -> Trả lời câu hỏi.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from Totext2 import BASE_DIR, OUTPUT_TEXT_FOLDER, EMBEDDING_MODEL

CHROMA_DB_DIR = BASE_DIR / ".chroma_db_method2"
LLM_MODEL = "gemini-3.5-flash"

load_dotenv(BASE_DIR / ".env")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class ChartAwareRAG:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY missing.")
            
        print("[Method 2 - Online] Đang nạp Vector Engine và LLM...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=api_key, temperature=0.2)

    def build_vector_store(self):
        txt_files = list(OUTPUT_TEXT_FOLDER.glob("*.txt"))
        if not txt_files:
            print("Thư mục trống. Hãy chạy offline_processor_method2.py trước!")
            return None

        print(f"Đang băm nhỏ và nạp {len(txt_files)} tệp dữ liệu làm giàu vào Chroma DB...")
        docs = []
        for txt_file in txt_files:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        chunks = text_splitter.split_documents(docs)

        if CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        print(f"-> Thành công! Đã lập chỉ mục xong {len(chunks)} Chunks dữ liệu.")

    def ask(self, query: str):
        if not hasattr(self, 'retriever'):
            raise RuntimeError("Vui lòng khởi tạo kho lưu trữ vector trước.")

        print("Đang thực hiện truy xuất thông tin chéo...")
        retrieved_chunks = self.retriever.invoke(query)
        
        # In log ra terminal để phục vụ viết phần phân tích định tính trong paper
        print(f"  -> Tìm thấy {len(retrieved_chunks)} Chunks liên quan. Đang cấu trúc hóa Prompt...")
        
        text_context = "\n\n".join([doc.page_content for doc in retrieved_chunks])

        prompt = f"""Bạn là một hệ thống AI phân tích tài chính thông minh phục vụ thị trường chứng khoán Việt Nam (VN-FinRAG). 
        Nhiệm vụ của bạn là sử dụng thông tin văn bản kết hợp với các ma trận dữ liệu biểu đồ chuyên sâu dưới đây để trả lời câu hỏi của nhà đầu tư một cách khách quan, chính xác nhất.
        Khi trích dẫn số liệu từ đồ thị, hãy chỉ rõ tên biểu đồ và các mốc thời gian tăng trưởng.

        [NGUỒN DỮ LIỆU ĐÃ TRÍCH XUẤT]:
        {text_context}

        [CÂU HỎI CỦA NHÀ ĐẦU TƯ]: {query}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

if __name__ == "__main__":
    try:
        rag = ChartAwareRAG()
        rag.build_vector_store()
        
        if hasattr(rag, 'retriever'):
            print("\n" + "="*50)
            print("[HỆ THỐNG TRUY VẤN BIỂU ĐỒ PHƯƠNG PHÁP 2 ĐÃ SẴN SÀNG]")
            print("="*50)
            while True:
                q = input("\nNhập câu hỏi phân tích của bạn (gõ 'exit' để dừng): ").strip()
                if q.lower() in ['exit', 'quit', 'q']:
                    break
                if not q: continue
                    
                try:
                    answer = rag.ask(q)
                    print("\n" + "🌟"*25)
                    print("BÁO CÁO PHÂN TÍCH KẾT QUẢ:")
                    print("🌟"*25)
                    print(answer)
                    print("🌟"*25)
                except Exception as e:
                    print(f"Lỗi truy vấn: {e}")
    except Exception as e:
        print(f"Lỗi khởi động hệ thống: {e}")