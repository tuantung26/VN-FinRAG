import os
import cv2
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
# pyrefly: ignore [missing-import]
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from llm import *
import fitz  # pymupdf
import tempfile

load_dotenv()

IMAGE_DIR = os.getenv("IMAGE_DIR")

class VisionProcessor:
    def __init__(self):
        print("set up models")
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        self.yolo_model = YOLOv10(model_path)
        self.deplot_processor = Pix2StructProcessor.from_pretrained("google/deplot")
        self.deplot_model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

    def crop_charts_from_folder(self, folder_path: str, IMAGE_DIR: str) -> list[str]:
        """dung yolo de crop anh"""
        counter = 1
        pages = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

        for page in pages:
            pdf_path = os.path.join(folder_path, page)
            # Dùng pymupdf render PDF sang ảnh, không cần Poppler
            doc = fitz.open(pdf_path)
            pix = doc[0].get_pixmap(dpi=150)
            doc.close()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            pix.save(tmp_path)

            result = self.yolo_model.predict(tmp_path, imgsz=1024, conf=0.2, verbose=False)
            for r in result:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.yolo_model.names[cls_id]
                    if label in ("figure", "table"):  # DocLayout-YOLO classes
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        img = cv2.imread(tmp_path)
                        cropped = img[y1:y2, x1:x2]
                        save_path = os.path.join(IMAGE_DIR, f"chart_{counter}.png")
                        cv2.imwrite(save_path, cropped)
                        counter += 1

            os.remove(tmp_path)

    
    def extract_tabular_data(self, image_path: str) -> str:
        """dung deplot de dich bang"""
        try:
            image = Image.open(image_path).convert("RGB")
            # Pix2Struct: header_text được render trực tiếp lên ảnh
            image_inputs = self.deplot_processor.image_processor(
                images=image,
                header_text="Generate underlying data table of the figure below:",
                return_tensors="pt"
            )
            predictions = self.deplot_model.generate(**image_inputs, max_new_tokens=512)
            return self.deplot_processor.tokenizer.decode(predictions[0], skip_special_tokens=True)

        except Exception as e:
            return f"loi o cai deplot {str(e)}"

    
    




    






    


