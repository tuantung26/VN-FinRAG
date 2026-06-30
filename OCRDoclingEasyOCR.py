import os
import torch
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

class OCR:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Document Converter on {device.upper()}...")
        
        # 1. Cấu hình OCR
        ocr_options = EasyOcrOptions()
        ocr_options.lang = ["vi", "en"]
        
        # 2. Khởi tạo Pipeline Options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options.device = device
        pipeline_options.ocr_options = ocr_options
        
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
        
        # Safe sort key for batch_X.pdf
        def get_batch_num(filename):
            name, _ = os.path.splitext(filename)
            parts = name.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                return int(parts[-1])
            return 999999
        pdf_files.sort(key=get_batch_num)

        for filename in pdf_files:
            file_path = os.path.join(folder_path, filename)
            text = self.ocr_pdf_with_docling(file_path)
            result_text += text + "\n\n"
            
        with open("result.txt", "w", encoding="utf-8") as file:
            file.write(result_text)
        print("--- Hoàn thành! ---")

if __name__ == "__main__":
    ocr_processor = OCR()
    ocr_processor.process_folder("batchs")