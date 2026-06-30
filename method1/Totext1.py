import sys
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# =====================================================================
# [CẤU HÌNH ĐƯỜNG DẪN Ở ĐÂY]
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent

# ĐƯỜNG DẪN ĐẦU VÀO (INPUT PATH)
PDF_FOLDER = BASE_DIR / "PDF" 

# CÁC THƯ MỤC ĐẦU RA
OUTPUT_TEXT_FOLDER = BASE_DIR / "Extracted_Texts"
OUTPUT_CHART_FOLDER = BASE_DIR / "Extracted_Charts"
PROCESSED_PDF_FOLDER = PDF_FOLDER / "Processed"

load_dotenv(BASE_DIR / ".env")

from docling_core.types.doc import TableItem, TextItem, SectionHeaderItem, PictureItem
from method1.OCRDoclingEasyOCR import OCR
from method1.SplitPDF import SplitPDFbyBatch

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

        print(f"\n[Offline] Processing: {input_path.name}")
        
        temp_batch_dir = BASE_DIR / f"temp_batches_{input_path.stem}"
        temp_batch_dir.mkdir(parents=True, exist_ok=True)
        
        doc_charts_dir = OUTPUT_CHART_FOLDER / input_path.stem
        doc_charts_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Splitting PDF into batches of 8 pages...")
        splitter = SplitPDFbyBatch(str(input_path), output_dir=str(temp_batch_dir), batch_size=8)
        batch_paths = splitter.split()
        
        print("Initializing Object Detection & OCR processor...")
        ocr_processor = OCR(device=self.device)
        
        batch_texts = []
        image_counter = 1 
        
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
                            
                            # ========================================================
                            # CÁCH 1: ÉP THẺ ẢNH VÀO ĐOẠN TEXT GẦN NHẤT
                            # ========================================================
                            img_tag = f" [CHART_IMAGE: {img_path.resolve().as_posix()}]"
                            if len(parts) > 0:
                                # Cộng dồn thẻ ảnh vào đoạn text/bảng/header liền trước nó
                                parts[-1] += img_tag
                            else:
                                # Nếu ảnh nằm ngay đầu trang (chưa có text), đành tạo phần tử mới
                                parts.append(img_tag.strip())
                            # ========================================================
                                
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
        print(f"  -> Output text saved: {out_path.name}")
        print(f"  -> Extracted {image_counter - 1} charts to {doc_charts_dir.name}/")

        return full_text

    def process_all(self):
        pdfs = sorted(PDF_FOLDER.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {PDF_FOLDER}. Vui lòng copy file PDF vào thư mục này!")
            return

        for pdf in pdfs:
            try:
                self.process(pdf)
                shutil.move(str(pdf), str(PROCESSED_PDF_FOLDER / pdf.name))
                print(f"Moved {pdf.name} to Processed folder.")
            except Exception as e:
                print(f"  ERROR processing {pdf.name}: {e}")

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_all()