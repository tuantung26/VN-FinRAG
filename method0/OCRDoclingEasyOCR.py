import os
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

class OCR:
    def __init__(self):
        print("Initializing Document Converter...")
        
        # 1. Cấu hình OCR
        ocr_options = EasyOcrOptions()
        ocr_options.lang = ["vi", "en"]
        
        # 2. Khởi tạo Pipeline Options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = ocr_options
        # FIX: Tắt các AI model dùng HuggingFace transformers (TableFormer, LayoutLM...)
        # để tránh lỗi "Could not import module 'AutoProcessor'"
        pipeline_options.do_table_structure = False
        
        # 3. Sử dụng PdfFormatOption để bao bọc pipeline_options (Đây là bước fix lỗi)
        self.docling = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def ocr_pdf_with_docling(self, pdf_path: str) -> str:
        print(f"Processing {pdf_path}...")
        result = self.docling.convert(pdf_path)
        return result.document.export_to_markdown()

    def process_folder(self, folder_path: str):
        result_text = ""
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        
        # FIX 1: Sort an toàn hơn
        def safe_sort_key(filename):
            parts = filename.split('_')
            if len(parts) > 1:
                try:
                    return int(parts[1].split('.')[0])
                except ValueError:
                    pass
            return filename  # fallback: sort theo tên
        
        pdf_files.sort(key=safe_sort_key)

        for filename in pdf_files:
            file_path = os.path.join(folder_path, filename)
            try:  # FIX 3: Bắt lỗi từng file
                text = self.ocr_pdf_with_docling(file_path)
                result_text += text + "\n\n"
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý {filename}: {e}")

        # FIX 2: Lưu output cùng folder với script
        output_path = os.path.join(os.path.dirname(__file__), "result.txt")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(result_text)
        print("--- Hoàn thành! ---")


if __name__ == "__main__":
    ocr_processor = OCR()
    ocr_processor.process_folder("pages")