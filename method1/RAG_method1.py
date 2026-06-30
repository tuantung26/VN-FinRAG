#!/usr/bin/env python3
"""
INTEGRATED ONLINE MULTIMODAL RAG PIPELINE (METHOD 1)
"""

import os
import sys
import re
import base64
import shutil
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Ensure stdout and stderr use UTF-8 on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Configure Paths
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PDF_FOLDER = BASE_DIR / "PDF"
OUTPUT_TEXT_FOLDER = BASE_DIR / "Extracted_Texts"
OUTPUT_CHART_FOLDER = BASE_DIR / "Extracted_Charts"
PROCESSED_PDF_FOLDER = PDF_FOLDER / "Processed"
CHROMA_DB_DIR = BASE_DIR / ".chroma_db_method1"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
FORCE_REINDEX = False  # Set to True to force reprocessing of texts and charts

load_dotenv(BASE_DIR / ".env")

# Import split and OCR libraries
from SplitPDF import SplitPDFbyBatch
from OCRDoclingEasyOCR import OCR

# --- Phase 1: Document Processing ---
class DocumentProcessor:
    def __init__(self, device: str = "auto"):
        self.device = device
        OUTPUT_TEXT_FOLDER.mkdir(parents=True, exist_ok=True)
        OUTPUT_CHART_FOLDER.mkdir(parents=True, exist_ok=True)
        PROCESSED_PDF_FOLDER.mkdir(parents=True, exist_ok=True)
        PDF_FOLDER.mkdir(parents=True, exist_ok=True)

    def process(self, input_path: Path) -> str:
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        print(f"\n[OCR] Processing: {input_path.name}")
        
        temp_batch_dir = BASE_DIR / f"temp_batches_{input_path.stem}"
        temp_batch_dir.mkdir(parents=True, exist_ok=True)
        
        doc_charts_dir = OUTPUT_CHART_FOLDER / input_path.stem
        doc_charts_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Splitting PDF into batches of 8 pages...")
        splitter = SplitPDFbyBatch(str(input_path), output_dir=str(temp_batch_dir), batch_size=8)
        batch_paths = splitter.split()
        
        print("  Initializing EasyOCR processor...")
        ocr_processor = OCR(device=self.device)
        
        batch_texts = []
        image_counter = 1 
        
        from docling_core.types.doc import TableItem, TextItem, SectionHeaderItem, PictureItem

        for i, batch_path in enumerate(batch_paths, 1):
            print(f"  Batch {i}/{len(batch_paths)}...")
            conv_result = ocr_processor.convert_pdf(batch_path)
            doc = conv_result.document
            
            parts = []
            for item, _ in doc.iterate_items():
                if isinstance(item, SectionHeaderItem):
                    parts.append(f"\n## {item.text}")
                elif isinstance(item, TextItem):
                    text = item.text.strip()
                    if text:
                        parts.append(text)
                elif isinstance(item, TableItem):
                    try:
                        md_table = item.export_to_markdown()
                        if md_table.strip():
                            parts.append(md_table)
                    except Exception:
                        pass
                elif isinstance(item, PictureItem):
                    try:
                        pil_img = item.image.pil_image
                        if pil_img is not None and pil_img.width > 150 and pil_img.height > 150:
                            img_path = doc_charts_dir / f"chart_{image_counter}.png"
                            pil_img.save(img_path)
                            
                            img_tag = f" [CHART_IMAGE: {img_path.resolve().as_posix()}]"
                            if len(parts) > 0:
                                parts[-1] += img_tag
                            else:
                                parts.append(img_tag.strip())
                            image_counter += 1
                    except Exception as e:
                        print(f"    Warning: could not process image: {e}")
                        
            batch_text = "\n\n".join(parts)
            if batch_text.strip():
                batch_texts.append(batch_text)
                
            Path(batch_path).unlink(missing_ok=True)
                
        try:
            temp_batch_dir.rmdir()
        except:
            pass
            
        full_text = "\n\n".join(batch_texts)

        out_path = OUTPUT_TEXT_FOLDER / f"{input_path.stem}.txt"
        out_path.write_text(full_text, encoding="utf-8")
        print(f"  -> Saved extracted text: {out_path.name}")
        print(f"  -> Extracted {image_counter - 1} charts to {doc_charts_dir.name}/")

        return full_text

    def process_all(self):
        pdfs = sorted(PDF_FOLDER.glob("*.pdf"))
        if not pdfs:
            print(f"No new PDFs found in {PDF_FOLDER}. Skipping OCR phase.")
            return

        for pdf in pdfs:
            try:
                self.process(pdf)
                shutil.move(str(pdf), str(PROCESSED_PDF_FOLDER / pdf.name))
                print(f"Moved {pdf.name} to Processed folder.")
            except Exception as e:
                print(f"  ERROR processing {pdf.name}: {e}")

# --- Helper function for source filtering ---
def map_doc_source_to_stem(doc_source: str) -> str:
    doc_source = str(doc_source).strip()
    doc_source_lower = doc_source.lower()
    if "ngan hang" in doc_source_lower or "ngân hàng" in doc_source_lower:
        return "Bao Cao Ngan Hang 06_2026"
    elif "bsr" in doc_source_lower or "doanh nghiep" in doc_source_lower or "doanh nghiệp" in doc_source_lower:
        return "Bao cao doanh nghiep BSR 3_2026(1)"
    elif "lop tai san" in doc_source_lower or "lớp tài sản" in doc_source_lower:
        return "Bao Cao Cac Lop Tai San 06_2026"
    elif "trai phieu" in doc_source_lower or "trái phiếu" in doc_source_lower:
        return "Bao Cao Trai Phieu - 06_2026"
    return doc_source

# --- Phase 2: Multimodal RAG Pipeline ---
class MultiModalRAG:
    def __init__(self, api_key: str):
        self.api_key = api_key
        print("[RAG] Initializing Embeddings and Gemini LLM...")
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=self.api_key, temperature=0.1)
        self.chart_pattern = re.compile(r"\[CHART_IMAGE:\s*(.+?)\]")
        self.vector_store = None

    def build_vector_store(self, force_reindex: bool = False):
        is_db_exists = CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir())
        new_pdfs = list(PDF_FOLDER.glob("*.pdf"))

        if is_db_exists and not force_reindex and not new_pdfs:
            print("\n[PHÁT HIỆN CACHE] Vector Database cũ đã được khởi tạo từ trước!")
            print(f"-> Hệ thống tự động nạp cơ sở dữ liệu từ thư mục: {CHROMA_DB_DIR}")
            from langchain_chroma import Chroma
            self.vector_store = Chroma(
                persist_directory=str(CHROMA_DB_DIR),
                embedding_function=self.embeddings
            )
        else:
            if force_reindex and CHROMA_DB_DIR.exists():
                print("\n[ÉP BUỘC RE-INDEX] Đang xóa cơ sở dữ liệu cũ theo yêu cầu...")
                try:
                    shutil.rmtree(CHROMA_DB_DIR)
                except Exception as e:
                    print(f"Cảnh báo khi dọn cache DB: {e}")

            # Run OCR on any new PDFs first
            processor = DocumentProcessor()
            processor.process_all()

            txt_files = list(OUTPUT_TEXT_FOLDER.glob("*.txt"))
            if not txt_files:
                raise ValueError(f"Không có dữ liệu văn bản nào được trích xuất tại {OUTPUT_TEXT_FOLDER}")

            print(f"[RAG] Loading {len(txt_files)} text files into Chroma...")
            docs = []
            from langchain_community.document_loaders import TextLoader
            for txt_file in txt_files:
                loader = TextLoader(str(txt_file), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = txt_file.stem
                docs.extend(loaded)

            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            print(f"Tổng số text chunks được lập chỉ mục: {len(chunks)}")

            from langchain_chroma import Chroma
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(CHROMA_DB_DIR)
            )
            print("[RAG] Vector store built successfully!")

    def encode_image_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def ask(self, query: str, doc_source: str = None) -> str:
        if self.vector_store is None:
            raise RuntimeError("You must build_vector_store() before asking questions.")

        filter_dict = None
        if doc_source:
            stem = map_doc_source_to_stem(doc_source)
            filter_dict = {"source": stem}
            print(f"  [RAG Filter] Querying metadata source: '{stem}'")
            
        retrieved_chunks = self.vector_store.similarity_search(query, k=4, filter=filter_dict)
        
        text_context = ""
        image_paths = []

        for doc in retrieved_chunks:
            chunk_text = doc.page_content
            found_images = self.chart_pattern.findall(chunk_text)
            image_paths.extend(found_images)
            
            clean_text = self.chart_pattern.sub("", chunk_text)
            text_context += clean_text + "\n"

        prompt_text = (
            "You are an expert financial analyst. Answer the question based on the provided financial context and charts (if any) below.\n"
            "Your answer MUST be a single, concise sentence. Do not use any newlines, paragraph breaks, or bullet points.\n"
            "If the context doesn't contain enough information, answer based on your knowledge but explicitly state that it is not in the document in Vietnamese language. "
            "If the context is not enough to answer, please return \"Tôi không biết\".\n\n"
            f"Context:\n{text_context.strip()}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        from langchain_core.messages import HumanMessage
        message_content = [
            {
                "type": "text", 
                "text": prompt_text
            }
        ]

        unique_image_paths = set(image_paths)
        for img_path in unique_image_paths:
            try:
                p = Path(img_path.strip())
                if not p.is_absolute():
                    p = BASE_DIR / p
                
                if p.exists():
                    b64_img = self.encode_image_base64(str(p))
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                    })
                    print(f"    -> Attached chart image to prompt: {p.name}")
                else:
                    print(f"    -> Warning: Chart image not found at {p}")
            except Exception as e:
                print(f"    -> Warning: Could not process chart image {img_path}: {e}")

        response = self.llm.invoke([HumanMessage(content=message_content)])
        answer = response.content
        if answer:
            answer = answer.replace("\r", "").replace("\n", " ").strip()
            answer = " ".join(answer.split())
        return answer

def main():
    api_key = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", "")).strip().strip('"').strip("'")
    if not api_key:
        print("Error: Bạn cần cấu hình GOOGLE_API_KEY hoặc GEMINI_API_KEY trong .env để chạy mô hình.")
        sys.exit(1)

    # Initialize RAG Pipeline
    rag = MultiModalRAG(api_key=api_key)
    rag.build_vector_store(force_reindex=FORCE_REINDEX)

    # Load and process benchmark
    method0_csv = Path(r"D:\DSR\project\method0\benchmark.csv")
    method1_csv = Path(r"D:\DSR\project\method1\benchmark.csv")
    
    if method1_csv.exists():
        print(f"\n[Eval] Loading existing benchmark CSV from {method1_csv}...")
        df = pd.read_csv(method1_csv)
    else:
        print(f"\n[Eval] Benchmark file in method1 not found. Loading from {method0_csv} and creating {method1_csv}...")
        if not method0_csv.exists():
            print(f"Error: Could not find benchmark CSV in method0 at {method0_csv}!")
            sys.exit(1)
        df = pd.read_csv(method0_csv)
        # Clear the model_answer column so that method1 will generate its own answers from scratch
        df['model_answer'] = pd.NA
        df.to_csv(method1_csv, index=False, encoding='utf-8-sig')

    print(f"\nBắt đầu quy trình kiểm tra và chạy Batch Inference cho {len(df)} hàng...")
    count_filled = 0

    for idx, row in df.iterrows():
        query = str(row.get('user_query', '')).strip()
        doc_src = row.get('doc_source')
        current_answer = str(row.get('model_answer', '')).strip()

        # Check if the answer is empty or placeholder
        if current_answer and current_answer not in ["nan", "<NA>", ".", "EMPTY_QUERY"]:
            print(f" -> [{idx + 1}/{len(df)}] BỎ QUA: Hàng này đã có câu trả lời từ trước.")
            continue

        if query and query not in ["nan", ""]:
            print(f" -> [{idx + 1}/{len(df)}] ĐANG XỬ LÝ [Source: {doc_src}]: {query[:55]}...")
            try_count = 0
            max_retries = 3
            success = False
            while try_count < max_retries:
                try:
                    answer = rag.ask(query, doc_source=doc_src)
                    df.at[idx, 'model_answer'] = answer
                    count_filled += 1
                    
                    # Save immediately after each answer to prevent data loss
                    df.to_csv(method1_csv, index=False, encoding='utf-8-sig')
                    time.sleep(5)
                    success = True
                    break
                except Exception as e:
                    err_msg = str(e)
                    print(f"    [LỖI DÒNG {idx + 1}] Gặp lỗi: {err_msg}")
                    
                    # Expired or Invalid API Key Detection
                    is_key_error = (
                        "API_KEY_INVALID" in err_msg or 
                        "API key not valid" in err_msg or 
                        "API key expired" in err_msg or
                        "invalid api key" in err_msg.lower() or
                        "400" in err_msg and "key" in err_msg.lower() or
                        "403" in err_msg
                    )
                    
                    if is_key_error:
                        print("    [API KEY ERROR] Phát hiện API key hết hạn hoặc không hợp lệ! Dừng tiến trình ngay lập tức.")
                        sys.exit(1)
                    
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        try_count += 1
                        print(f"    [429 RATE LIMIT] Thử lại lần {try_count}/{max_retries} sau 30 giây...")
                        time.sleep(30)
                    else:
                        # Other random exceptions
                        df.at[idx, 'model_answer'] = f"Error: {err_msg}"
                        df.to_csv(method1_csv, index=False, encoding='utf-8-sig')
                        break
            if not success:
                # If retries exceeded, stop processing this query (leave it for next runs)
                break
        else:
            df.at[idx, 'model_answer'] = "EMPTY_QUERY"
            df.to_csv(method1_csv, index=False, encoding='utf-8-sig')

    print(f"\n[QUY TRÌNH HOÀN THÀNH]")
    print(f"-> Số lượng hàng vừa được điền mới thành công: {count_filled}")
    print(f"-> Toàn bộ dữ liệu cập nhật đã được ghi đè an toàn tại: {method1_csv}")

if __name__ == "__main__":
    main()
