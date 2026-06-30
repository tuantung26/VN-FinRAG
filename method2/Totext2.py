#!/usr/bin/env python3
"""
FILE 1: METHOD 2 - CHART-AWARE INDEXING WITH LANGCHAIN STRUCTURED OUTPUT (OFFLINE)
Nhiệm vụ: Cắt PDF -> Trích xuất ảnh biểu đồ -> DePlot lấy dữ liệu thô 
-> LangChain Structured Output (Pydantic) ép Gemini phân tích cấu trúc 
-> Nhúng chuỗi dữ liệu chuẩn hóa vào file Text.
"""

import sys
import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
# =====================================================================
# [CẤU HÌNH ĐƯỜNG DẪN ĐẦU VÀO VÀ ĐẦU RA]
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "PDF"  # ---> THƯ MỤC ĐẦU VÀO CHỨA FILE PDF
OUTPUT_TEXT_FOLDER = BASE_DIR / "Extracted_Texts_Method2"
PROCESSED_PDF_FOLDER = PDF_FOLDER / "Processed_Method2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Optional flag to skip DePlot heavy processing for faster runs (set via env var)
SKIP_DEEP_PROCESSING = os.getenv("SKIP_DEEP_PROCESSING", "False").lower() == "true"

load_dotenv(BASE_DIR / ".env")

import torch
from pydantic import BaseModel, Field
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from docling_core.types.doc import TableItem, TextItem, SectionHeaderItem, PictureItem
from OCRDoclingEasyOCR import OCR
from SplitPDF import SplitPDFbyBatch

# =====================================================================
# ĐỊNH NGHĨA PYDANTIC SCHEMA CHO BIỂU ĐỒ (STRUCTURED OUTPUT)
# =====================================================================
class DataPoint(BaseModel):
    label: str = Field(description="Nhãn danh mục, mốc thời gian, hoặc giá trị trục X (ví dụ: 'Q1/2024', 'Năm 2025', 'Dầu Diesel')")
    value: str = Field(description="Giá trị số chính xác được trích xuất từ biểu đồ (giữ nguyên định dạng gốc)")
    unit: Optional[str] = Field(default=None, description="Đơn vị đo lường nếu có (ví dụ: 'tỷ VND', '%', 'thùng/ngày')")
    series: Optional[str] = Field(default=None, description="Tên chuỗi dữ liệu trong biểu đồ nhiều cột/đường (ví dụ: 'Doanh thu', 'Kế hoạch')")


class ChartAnalysisResult(BaseModel):
    chart_title: str = Field(description="Tiêu đề biểu đồ hoặc mô tả ngắn gọn nội dung biểu đồ nếu không thấy tiêu đề")
    chart_type: str = Field(description="Phân loại biểu đồ (ví dụ: 'Bar Chart', 'Line Chart', 'Pie Chart', 'Stacked Bar Chart')")
    data_matrix: List[DataPoint] = Field(description="Mảng dữ liệu số liệu chi tiết cấu trúc hóa được trích xuất từ biểu đồ")
    financial_insights: List[str] = Field(description="Danh sách tối thiểu 3 câu nhận định phân tích tài chính chuyên sâu về xu hướng, tăng trưởng YoY/QoQ, các điểm đỉnh/đáy đột biến")
    x_axis: str = Field(default="Unknown", description="Mô tả trục X của biểu đồ, nếu có")
    y_axis: str = Field(default="Unknown", description="Mô tả trục Y của biểu đồ, nếu có")
    legend: List[str] = Field(default_factory=list, description="Danh sách các nhãn legend nếu có")
    detected_categories: List[str] = Field(default_factory=list, description="Các danh mục hoặc series được phát hiện trong biểu đồ")
    extracted_data_table: str = Field(default="", description="Bảng dữ liệu số dạng markdown")
    trend_analysis: str = Field(default="Unknown", description="Mô tả xu hướng tổng quan của biểu đồ")
    important_observations: List[str] = Field(default_factory=list, description="Những quan sát quan trọng, bất thường, so sánh")
    natural_language_summary: str = Field(default="Unknown", description="Tóm tắt tự nhiên ngắn gọn của biểu đồ")
    keywords: List[str] = Field(default_factory=list, description="Các từ khóa liên quan, 3‑7 từ")


# =====================================================================
# CLASS XỬ LÝ VÀ PHÂN TÍCH BIỂU ĐỒ
# =====================================================================
class ChartSemanticExtractor:
    def __init__(self, device: str = "auto"):
        if device == "cuda":
            self.device = "cuda"
        elif device == "cpu":
            self.device = "cpu"
        else:  # auto
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.info(f"[DePlot] Loading Google DePlot model on {self.device.upper()}")
        self.processor = Pix2StructProcessor.from_pretrained('google/deplot')

        # Optimize VRAM using float16 on CUDA
        if self.device == "cuda":
            self.model = Pix2StructForConditionalGeneration.from_pretrained(
                'google/deplot',
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to(self.device)
        
        # Khởi tạo LangChain với Mô hình Cấu trúc đầu ra
        raw_keys = os.getenv("GOOGLE_API_KEYS")
        if not raw_keys:
            raise ValueError("GOOGLE_API_KEYS not found in .env. Provide at least one API key.")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not self.api_keys:
            raise ValueError("No valid API keys provided in GOOGLE_API_KEYS.")
        self.current_key_index = 0
        self._init_gemini_client()

        # Prompt template for financial chart analysis
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một chuyên gia phân tích tài chính cấp cao. Nhiệm vụ của bạn là nhận bảng dữ liệu thô từ biểu đồ và trả về một đối tượng cấu trúc đầy đủ bao gồm các trường: chart_title, chart_type, x_axis, y_axis, legend, detected_categories, extracted_data_table, trend_analysis, important_observations, financial_insights, natural_language_summary, keywords. Nếu không chắc chắn, trả về \"Unknown\" hoặc danh sách rỗng.")
            ,
            ("human", "Dưới đây là bảng dữ liệu thô trích xuất từ biểu đồ:\n\n{raw_data_table}\n\nHãy phân tích và điền vào các trường trên. Đảm bảo trả về dữ liệu cho mọi trường, ngay cả khi không có thông tin (sử dụng \"Unknown\" hoặc danh sách rỗng).")
        ])

        # Assemble the chain
        self.analysis_chain = self.prompt_template | self.structured_chain
    def _init_gemini_client(self):
        """Initialize Gemini client with the current API key for structured output."""
        api_key = self.api_keys[self.current_key_index]
        self.raw_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key, temperature=0.1)
        self.structured_chain = self.raw_llm.with_structured_output(ChartAnalysisResult)


    def extract(self, pil_image, chart_id: str) -> str:
        """Receive a PIL image and a chart identifier, return a rich markdown block containing structured analysis.
        The method uses DePlot to generate a raw data table, then invokes the Gemini structured-output chain.
        All fields from ChartAnalysisResult are rendered, with placeholders for missing values.
        """
        if SKIP_DEEP_PROCESSING:
            return "\n\n[CHART_SKIPPED: Phân tích biểu đồ bị bỏ qua theo cấu hình]\n\n"
        try:
            # 1. Generate raw data table via DePlot
            logging.info("    [DePlot] Generating raw table...")
            raw_inputs = self.processor(
                images=pil_image,
                text="Generate underlying data table of the figure below:",
                return_tensors="pt",
            )
            # Move tensors to the appropriate device
            if self.device == "cuda":
                inputs = {k: v.half().to(self.device) if isinstance(v, torch.Tensor) else v for k, v in raw_inputs.items()}
            else:
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in raw_inputs.items()}

            predictions = self.model.generate(**inputs, max_new_tokens=1024, max_length=None)
            raw_table_str = self.processor.decode(predictions[0], skip_special_tokens=True)
            clean_table_str = raw_table_str.replace("\u003c0x0A\u003e", "\n")

            # 2. Structured analysis via Gemini
            logging.info("    [Gemini] Analyzing structure and insights...")
            chart_data: ChartAnalysisResult = self.analysis_chain.invoke({"raw_data_table": clean_table_str})

            # Helper to format data matrix as markdown table
            def _format_data_table(matrix: List[DataPoint]) -> str:
                if not matrix:
                    return ""
                header = "| Series | Label | Value | Unit |"
                separator = "|---|---|---|---|"
                rows = []
                for dp in matrix:
                    series = dp.series or "Data"
                    unit = dp.unit or ""
                    rows.append(f"| {series} | {dp.label} | {dp.value} | {unit} |")
                return "\n".join([header, separator] + rows)

            # Serialize all fields
            data_table_md = chart_data.extracted_data_table or _format_data_table(chart_data.data_matrix)
            insights_formatted = "\n".join([f"- {ins}" for ins in chart_data.financial_insights])
            observations_formatted = "\n".join([f"- {obs}" for obs in chart_data.important_observations])
            keywords_formatted = ", ".join(chart_data.keywords)

            enriched_text = (
                f"---\n"
                f"Chart ID: {chart_id}\n"
                f"Chart Title: {chart_data.chart_title}\n"
                f"Chart Type: {chart_data.chart_type}\n"
                f"X-axis: {chart_data.x_axis}\n"
                f"Y-axis: {chart_data.y_axis}\n"
                f"Legend: {', '.join(chart_data.legend) if chart_data.legend else 'Unknown'}\n"
                f"Detected Categories: {', '.join(chart_data.detected_categories) if chart_data.detected_categories else 'Unknown'}\n"
                f"Extracted Numerical Data:\n{data_table_md}\n"
                f"Trend Analysis:\n{chart_data.trend_analysis}\n"
                f"Important Observations:\n{observations_formatted if observations_formatted else 'None'}\n"
                f"Financial Insights:\n{insights_formatted}\n"
                f"Natural Language Summary:\n{chart_data.natural_language_summary}\n"
                f"Keywords: {keywords_formatted if keywords_formatted else 'None'}\n"
                f"---\n"
            )
            return enriched_text
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower() or "429" in err_msg:
                print(f"\n[CRITICAL ERROR] Hết quota/token API Gemini! Dừng chương trình để thay API Key. Chi tiết: {e}")
                raise e
            print(f"    Warning: Lỗi phân tích cấu trúc biểu đồ: {e}")
            return "\n\n[CHART_ERROR: Không thể giải mã dữ liệu đồ họa]\n\n"

# =====================================================================
# BIÊN DỊCH VÀ PIPELINE TỔNG THỂ
# =====================================================================
class DocumentProcessor:
    def __init__(self, device: str = "auto"):
        self.device = device
        OUTPUT_TEXT_FOLDER.mkdir(parents=True, exist_ok=True)
        PROCESSED_PDF_FOLDER.mkdir(parents=True, exist_ok=True)
        PDF_FOLDER.mkdir(parents=True, exist_ok=True)
        
        self.extractor = ChartSemanticExtractor(device=device)

    def process(self, input_path: Path) -> str:
        print(f"\n[Offline] Processing: {input_path.name}")
        temp_batch_dir = BASE_DIR / f"temp_batches_{input_path.stem}"
        temp_batch_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            splitter = SplitPDFbyBatch(str(input_path), output_dir=str(temp_batch_dir), batch_size=8)
            batch_paths = splitter.split()
            
            ocr_processor = OCR(device=self.device)
            batch_texts = []
            chart_count = 0
            
            out_path = OUTPUT_TEXT_FOLDER / f"{input_path.stem}.txt"
            # Khởi tạo file trống để cập nhật liên tục
            out_path.write_text("", encoding="utf-8")  # initialize empty file
            
            for i, batch_path in enumerate(batch_paths, 1):
                try:
                    logging.info(f"Batch {i}/{len(batch_paths)} processing...")
                    conv_result = ocr_processor.convert_pdf(batch_path)
                    doc = conv_result.document
                    
                    parts = []
                    for item, _ in doc.iterate_items():
                        if isinstance(item, SectionHeaderItem):
                            parts.append(f"\n## {item.text}")
                        elif isinstance(item, TextItem):
                            text = item.text.strip()
                            if text: parts.append(text)
                        elif isinstance(item, TableItem):
                            try:
                                md_table = item.export_to_markdown()
                                if md_table.strip(): parts.append(md_table)
                            except Exception: pass
                        elif isinstance(item, PictureItem):
                            try:
                                pil_img = item.image.pil_image
                                if pil_img is not None and pil_img.width > 150 and pil_img.height > 150:
                                    chart_count += 1
                                    logging.info(f"-> Extracting structure for chart #{chart_count}...")

                                    # Run chart extraction (may be skipped based on flag)
                                    chart_id = f"{input_path.stem}_chart_{chart_count}"  # unique identifier for this chart
                                    semantic_text = self.extractor.extract(pil_img, chart_id)

                                    # Append or merge into the last text part to avoid fragmentation
                                    if parts:
                                        parts[-1] += semantic_text
                                    else:
                                        parts.append(semantic_text.strip())
                            except Exception as e:
                                if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower() or "429" in str(e):
                                    raise e
                                print(f"    Warning: Lỗi trích xuất ảnh từ PDF: {e}")
                                
                    batch_text = "\n\n".join(parts)
                    if batch_text.strip():
                        batch_texts.append(batch_text)
                        # Ghi ngay vào file để theo dõi thời gian thực
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(batch_text + "\n\n")
                finally:
                    # Đảm bảo xóa batch file ngay cả khi gặp lỗi trong lúc convert
                    Path(batch_path).unlink(missing_ok=True)
                    
            full_text = "\n\n".join(batch_texts)
            out_path.write_text(full_text, encoding="utf-8")
            logging.info(f"Completed enriched text file: {out_path.name}")
            return full_text
            
        finally:
            # Dọn dẹp thư mục tạm thời
            try:
                for f in temp_batch_dir.glob("*"):
                    f.unlink(missing_ok=True)
                temp_batch_dir.rmdir()
            except Exception as e:
                print(f"Warning: Không thể dọn dẹp thư mục tạm {temp_batch_dir}: {e}")

    def process_all(self):
        pdfs = sorted(PDF_FOLDER.glob("*.pdf"))
        if not pdfs:
            print(f"Không tìm thấy file PDF đầu vào nào tại {PDF_FOLDER}!")
            return
        for pdf in pdfs:
            try:
                self.process(pdf)
                shutil.move(str(pdf), str(PROCESSED_PDF_FOLDER / pdf.name))
                logging.info(f"Moved {pdf.name} to processed folder.")
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower() or "429" in str(e):
                    logging.critical(f"[CRITICAL STOP] Gemini quota exhausted: {e}")
                    raise e
                logging.error(f"Error processing {pdf.name}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Method 2 - Offline PDF Processor")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Thiết bị chạy mô hình (auto, cuda, cpu)")
    args = parser.parse_args()
    
    processor = DocumentProcessor(device=args.device)
    processor.process_all()