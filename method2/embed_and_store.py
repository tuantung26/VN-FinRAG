from milvusdb import get_collection
from SplitPDF import SplitPDF
import os
from dotenv import load_dotenv
from config import (
    COLLECTION_NAME, IMAGE_DIR, PAGES_DIR,
    RESULTS_FILE, MAX_PDF_TEXT_CHARS
)

load_dotenv()

from milvusdb import *
from imageProcess import *
from text_chunker import *
from embedding import *
from llm import *
from OCR import *
from SplitPDF import *

def process_and_store(document_path: str):
    collection = get_collection()
    vp = VisionProcessor()
    ocr = OCR()
    jina = Jina()

    os.makedirs(IMAGE_DIR, exist_ok=True)

    print("Xu li tai lieu")
    split_pdf = SplitPDF(document_path)
    split_pdf.splitPDF(PAGES_DIR)

    print("Xu li OCR")
    ocr.process_folder(PAGES_DIR, RESULTS_FILE)
    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        raw_text = file.read()

    print("Xu li chunk text")
    text_chunks = chunk_text_by_words(raw_text, MAX_PDF_TEXT_CHARS)
    
    print("Xu li embedding text")
    for chunk in text_chunks:
        vector = jina.EmbeddingBysentence(chunk)
        data = [{
            "chunk_type": "text",
            "vector": vector,
            "text": chunk,
            "image_path": "",
            "tabular_data": "",
        }]
        collection.insert(collection_name=COLLECTION_NAME, data=data)
    
    print("xu li anh")

    vp.crop_charts_from_folder(PAGES_DIR, IMAGE_DIR)
    for file in os.listdir(IMAGE_DIR):
        if file.lower().endswith('.png'):
            image_path = os.path.join(IMAGE_DIR, file)
            try:
                tabular_data = vp.extract_tabular_data_vlm(image_path)
                content = vp.get_image_content(image_path)
                vector = jina.EmbeddingBysentence(content)

                data = [{
                    "chunk_type": "chart",
                    "vector": vector,
                    "text": content,
                    "image_path": image_path,
                    "tabular_data": tabular_data,
                }]
                collection.insert(collection_name=COLLECTION_NAME, data=data)
                print(f"  [OK] {file}")
            except Exception as e:
                print(f"  [SKIP] {file}: {e}")

    collection.load_collection(collection_name=COLLECTION_NAME)
    print("Hoan Tat")


if __name__ == "__main__":
    process_and_store("D:\\personal\\tucode\\Advanced\\pdftest.pdf")