import os
from pypdf import PdfReader, PdfWriter

class SplitPDFbyBatch:
    def __init__(self, pdf_path: str, output_dir: str = "batchs", batch_size: int = 8):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.batch_size = batch_size

    def split(self) -> list[str]:
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        reader = PdfReader(self.pdf_path)
        os.makedirs(self.output_dir, exist_ok=True)

        total_pages = len(reader.pages)
        writer = PdfWriter()
        batch_paths = []

        for i, page in enumerate(reader.pages):
            writer.add_page(page)
            if (i + 1) % self.batch_size == 0 or (i + 1) == total_pages:
                batch_num = (i // self.batch_size) + 1
                output_path = os.path.join(self.output_dir, f"batch_{batch_num}.pdf")
                
                with open(output_path, "wb") as output:
                    writer.write(output)
                    
                print(f"Written batch {batch_num} to {output_path}")
                batch_paths.append(output_path)
                writer = PdfWriter()
                
        return batch_paths

if __name__ == "__main__":
    pdf_path = r"D:\FPT IS\OCR\Bao cao doanh nghiep BSR 3_2026(1) - Copy.pdf"
    splitter = SplitPDFbyBatch(pdf_path, output_dir="batchs", batch_size=8)
    splitter.split()

