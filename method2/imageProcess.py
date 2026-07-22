from llm import get_llm_wandb
import os
import cv2
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
# pyrefly: ignore [missing-import]
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from llm import *
import fitz  # pymupdf
import tempfile
from config import IMAGE_DIR
from llm import *

class VisionProcessor:
    def __init__(self):
        print("set up models")
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        self.llm = get_llm_wandb()
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


    def get_image_content(self,path: str):
        with open(path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        message = [
            HumanMessage(content = [
                {"type": "text", "text": "Hãy mô tả ngắn gọn biểu đồ sau bằng cách đưa ra những phần sau một cách đúng cấu trúc: nội dung của biểu đồ bao gồm việc biểu đồ chứa cái gì, cột x, cột y là gì nếu có, Thêm vào đó hãy đưa ra các insight cần có của biểu đồ này (Insight: insight 1, 2....)  (không dùng markdown):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ])
    ]
        response = self.llm.invoke(message)
        return response.content 

    
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

    
    def extract_tabular_data_vlm(self, image_path: str) -> str:

        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        message = [
            HumanMessage(content = [
                {"type": "text", "text": "đọc biểu đồ sau và xuất ra thông tin đầy đủ và chính xác ở dạng bảng của biểu đồ này ở định dạng markdown, chỉ cần đưa ra một cái bảng thôi (không dùng markdown):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ])
        ]
        response = self.llm.invoke(message)
        return response.content



