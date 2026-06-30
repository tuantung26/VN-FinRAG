from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_classic.chains import create_retrieval_chain
# pyrefly: ignore [missing-import]
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

from SplitPDF import SplitPDFbyBatch
from OCRDoclingEasyOCR import OCR

load_dotenv()
# Load multiple API keys for rotation
API_KEYS = []
API_INDEX_FILE = os.path.join(os.path.dirname(__file__), ".api_key_index")

def _load_api_keys():
    """Read GOOGLE_API_KEYS env variable (comma‑separated) or fallback to single key."""
    keys_str = os.getenv("GOOGLE_API_KEYS") or os.getenv("GOOGLE_API_KEY")
    if not keys_str:
        return []
    # Remove possible quotes and whitespace
    keys = [k.strip().strip('"').strip("'") for k in keys_str.split(',') if k.strip()]
    return keys

def _read_index():
    try:
        with open(API_INDEX_FILE, "r") as f:
            return int(f.read().strip())
    except Exception:
        return 0

def _write_index(idx: int):
    with open(API_INDEX_FILE, "w") as f:
        f.write(str(idx))

def get_current_api_key() -> str:
    global API_KEYS
    if not API_KEYS:
        API_KEYS = _load_api_keys()
    if not API_KEYS:
        raise RuntimeError("No API key found in environment.")
    idx = _read_index() % len(API_KEYS)
    return API_KEYS[idx]

def rotate_api_key():
    global API_KEYS
    if not API_KEYS:
        API_KEYS = _load_api_keys()
    if not API_KEYS:
        raise RuntimeError("No API key to rotate.")
    idx = (_read_index() + 1) % len(API_KEYS)
    _write_index(idx)
    return API_KEYS[idx]


# --- CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG CỦA TÙNG ---
PDF_FOLDER_PATH = r"D:\DSR\project\method0\PDF"   # Thư mục chứa các file PDF đầu vào
BENCHMARK_CSV_PATH = "benchmark2.csv"             # File dữ liệu test đầu vào
OUTPUT_CSV_PATH = "benchmark_results.csv"         # File kết quả đầu ra

# --- CẤU HÌNH KIỂM TRA VECTOR DATABASE ---
FORCE_REINDEX = False  # Đổi thành True nếu bạn muốn ÉP BUỘC CHẠY LẠI OCR từ đầu

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
COLLECTION_NAME = "pdf_rag_collection"
PERSIST_DIR = Path(__file__).resolve().parent / ".chroma_pdf_rag"


class FolderFileLoader:
    """Lớp xử lý quét folder và OCR tuần tự từng file một"""
    def __init__(self, folder_path: str | Path) -> None:
        self.folder_path = Path(folder_path)
    
    def load_all_pdfs(self) -> list[Document]:
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Thư mục không tồn tại: {self.folder_path}")
            
        pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"Cảnh báo: Không tìm thấy file PDF nào trong thư mục '{self.folder_path}'")
            return []
            
        print(f"Tìm thấy {len(pdf_files)} file PDF mới. Bắt đầu thực hiện quy trình OCR nâng cao...")
        all_documents = []
        ocr_engine = OCR()
        
        for idx, filename in enumerate(pdf_files):
            pdf_path = self.folder_path / filename
            print(f"\n[{idx + 1}/{len(pdf_files)}] Đang OCR tài liệu: {filename}")
            
            try:
                splitter = SplitPDFbyBatch(str(pdf_path))
                batch_paths = splitter.split()
                
                file_text = ""
                for b_path in batch_paths:
                    file_text += ocr_engine.ocr_pdf_with_docling(b_path) + "\n\n"
                
                all_documents.append(Document(page_content=file_text, metadata={"source": filename}))
                print(f"-> Hoàn thành OCR thành công cho file: {filename}")
            except Exception as e:
                print(f"-> [BỎ QUA FILE] Lỗi phân tích cú pháp {filename}: {e}")
                
        return all_documents


class RAGPipeline:
    def __init__(self, folder_path: str, api_key: str) -> None:
        self.folder_path = folder_path
        self.api_key = api_key
        self.embeddings = LCHuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.db = None
        self.chain = None

    def build(self) -> RAGPipeline:
        # Kiểm tra xem Vector Database cũ đã tồn tại và người dùng không ép buộc lập chỉ mục lại hay chưa
        is_db_exists = PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())
        
        if is_db_exists and not FORCE_REINDEX:
            print("\n[PHÁT HIỆN CACHE] Vector Database cũ đã được khởi tạo từ trước!")
            print(f"-> Hệ thống tự động nạp cơ sở dữ liệu từ thư mục: {PERSIST_DIR}")
            print("-> BỎ QUA HOÀN TOÀN BƯỚC OCR VÀ CHUNKING ĐỂ TIẾT KIỆM THỜI GIAN.")
            
            # Đọc trực tiếp dữ liệu từ thư mục chứa Chroma cũ lên bộ nhớ
            self.db = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(PERSIST_DIR)
            )
        else:
            if FORCE_REINDEX and PERSIST_DIR.exists():
                print("\n[ÉP BUỘC RE-INDEX] Đang xóa cơ sở dữ liệu cũ theo yêu cầu...")
                try:
                    shutil.rmtree(PERSIST_DIR)
                except Exception as e:
                    print(f"Cảnh báo khi dọn cache DB: {e}")

            print("\n[KHỞI TẠO MỚI] Cơ sở dữ liệu chưa tồn tại hoặc đã bị xóa.")
            
            # Thực hiện luồng xử lý OCR và Chunking tốn thời gian (chỉ chạy 1 lần đầu)
            docs = FolderFileLoader(self.folder_path).load_all_pdfs()
            if not docs:
                raise ValueError("Không có dữ liệu văn bản nào được trích xuất thành công từ thư mục PDF.")
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            print(f"\nTổng số text chunks được lập chỉ mục: {len(chunks)}")
            
            print("Đang embedding văn bản và ghi dữ liệu cố định vào lưu trữ của Chroma...")
            self.db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(PERSIST_DIR)
            )

        # Thiết lập cấu hình chuỗi kết nối xử lý sinh của LLM
        retriever = self.db.as_retriever(search_kwargs={"k": TOP_K})
        
        prompt_template = """You are an expert financial analyst. Use the provided financial context and optional chart image (base64) to answer the question.

Provide a clear, confident answer in a single sentence in Vietnamese. If the information is insufficient, respond with "Tôi không biết" or "Không đủ dữ liệu để trả lời". Do not add extra explanations, bullet points, or newlines.

Context:
{context}

Chart (base64):
{image_base64}

Question: {input}
Answer:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=self.api_key, temperature=0.1)
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.chain = create_retrieval_chain(retriever, question_answer_chain)
        return self

    def ask(self, question: str, image_base64: str = "") -> str:
        if self.chain is None:
            raise RuntimeError("Pipeline chưa được thiết lập. Vui lòng gọi build() trước.")
        # Pass both question and image data to the chain
        result = self.chain.invoke({"input": question, "image_base64": image_base64})
        answer = result.get("answer", "")
        if answer:
            # Loại bỏ các ký tự xuống dòng và khoảng trắng dư thừa để chèn đẹp vào CSV
            answer = answer.replace("\r", "").replace("\n", " ").strip()
            answer = " ".join(answer.split())
        return answer


def load_chart_base64(chartid: str) -> str:
    """Load chart image file by chartid and return base64 string, or empty string if not found."""
    if not chartid or pd.isna(chartid):
        return ""
    # Assume chartid includes file extension; if not, try common extensions
    possible_paths = []
    base_dir = os.path.join(os.path.dirname(__file__), "chart_image")
    possible_paths.append(os.path.join(base_dir, chartid))
    # Try adding .png or .jpg if missing
    if not os.path.splitext(chartid)[1]:
        possible_paths.append(os.path.join(base_dir, f"{chartid}.png"))
        possible_paths.append(os.path.join(base_dir, f"{chartid}.jpg"))
    for p in possible_paths:
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
    return ""

def main():
    # Obtain the first API key (rotation will happen on quota errors)
    api_key = get_current_api_key().strip()
    if not api_key:
        print("Error: Bạn cần cấu hình GOOGLE_API_KEYS trong file .env để chạy mô hình.")
        return

    # Khởi tạo kiến trúc pipeline thông minh (Sẽ nạp cache DB nếu đã chạy OCR trước đó)
    rag = RAGPipeline(folder_path=PDF_FOLDER_PATH, api_key=api_key).build()
    print("\n[HỆ THỐNG SẴN SÀNG] Đã sẵn sàng kết nối kho tri thức.")

    # Đọc tập dữ liệu Benchmark CSV đầu vào
    if not os.path.exists(BENCHMARK_CSV_PATH):
        print(f"Error: Không tìm thấy tệp benchmark định dạng CSV tại '{BENCHMARK_CSV_PATH}'")
        return

    df = pd.read_csv(BENCHMARK_CSV_PATH)
    if 'user_query' not in df.columns:
        print("Error: Tệp CSV đầu vào của bạn thiếu trường thông tin bắt buộc 'user_query'.")
        return

    # Nếu file CSV chưa bao giờ có cột model_answer, tự động tạo cột trống ban đầu
    if 'model_answer' not in df.columns:
        df['model_answer'] = pd.NA

    print(f"\nBắt đầu quy trình kiểm tra và chạy Batch Inference cho {len(df)} hàng...")
    
    count_filled = 0
    # Duyệt qua từng hàng theo chỉ mục (index) để cập nhật trực tiếp vào DataFrame
    for idx, row in df.iterrows():
        query = str(row['user_query']).strip()
        current_answer = str(row['model_answer']).strip()
        chartid = row.get('chartid')
        
        # KIỂM TRA: Nếu ô model_answer đã có chữ dữ liệu (không phải rỗng, nan, hoặc rỗng hoàn toàn)
        if current_answer and current_answer != "nan" and current_answer != "<NA>":
            print(f" -> [{idx + 1}/{len(df)}] BỎ QUA: Hàng này đã có câu trả lời từ trước.")
            continue
            
        # Nếu chưa có câu trả lời và query hợp lệ thì mới tiến hành gọi API LLM
        if query and query != "nan":
            print(f" -> [{idx + 1}/{len(df)}] ĐANG XỬ LÝ: {query[:55]}...")
            chart_base64 = load_chart_base64(str(chartid))
            try_count = 0
            max_retries = 3
            success = False
            while try_count < max_retries:
                try:
                    # Gọi RAG tìm kiếm context và sinh câu trả lời
                    answer = rag.ask(query, image_base64=chart_base64)
                    df.at[idx, 'model_answer'] = answer
                    count_filled += 1
                    
                    # Save progress after each answer
                    df.to_csv(BENCHMARK_CSV_PATH, index=False, encoding='utf-8-sig')
                    time.sleep(5)
                    success = True
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        # Rotate to a new API key and retry
                        try:
                            new_key = rotate_api_key()
                            rag = RAGPipeline(folder_path=PDF_FOLDER_PATH, api_key=new_key).build()
                            print("[API ROTATION] Switched to a new API key due to quota limit.")
                        except Exception as rot_err:
                            print(f"[API ROTATION FAILED] {rot_err}")
                            raise
                        try_count += 1
                        print(f"    [QUOTA LIMIT] Switched API key, retry {try_count}/{max_retries} after 30s. Details: {err_msg[:200]}")
                        time.sleep(30)
                    else:
                        print(f"    [ANSWER ERROR] {e}")
                        break
            
            if not success:
                break
        else:
            df.at[idx, 'model_answer'] = "EMPTY_QUERY"

    print(f"\n[QUY TRÌNH HOÀN THÀNH]")
    print(f"-> Số lượng hàng vừa được điền mới thành công: {count_filled}")
    print(f"-> Toàn bộ dữ liệu cập nhật đã được ghi đè an toàn tại: {BENCHMARK_CSV_PATH}")


if __name__ == "__main__":
    main()