import os
from pypdf import PdfReader, PdfWriter

class SplitPDF:
    def __init__(self, document_path):
        self.document_path = document_path
        self.reader = PdfReader(document_path)

    def splitPDF(self, output_path: str):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i, page in enumerate(self.reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            output_file = os.path.join(output_path, f"page_{i+1}.pdf")
            with open(output_file, "wb") as output:
                writer.write(output)
        
        return output_path


            
        
