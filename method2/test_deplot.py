"""
Test DePlot (Pix2Struct) - chart to table extraction
Chạy: python test_deplot.py
Hoặc: python test_deplot.py path/to/chart.png
"""
import sys
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

# ======== CONFIG ========
# Đường dẫn tới ảnh chart cần test
IMAGE_PATH = r"D:\personal\tucode\Advanced\Screenshot 2026-07-20 132347.png"
# ========================


def test_deplot(image_path: str):
    print("Đang load model DePlot (google/deplot)...")
    processor = Pix2StructProcessor.from_pretrained("google/deplot")
    model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
    print("Load xong!\n")

    print(f"Đang xử lý ảnh: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"  Kích thước ảnh: {image.size}")

    image_inputs = processor.image_processor(
        images=image,
        header_text="Generate underlying data table of the figure below:",
        return_tensors="pt"
    )

    print("Đang generate...")
    predictions = model.generate(**image_inputs, max_new_tokens=512)
    result = processor.tokenizer.decode(predictions[0], skip_special_tokens=True)

    print("\n========== KẾT QUẢ DEPLOT ==========")
    print(result)
    print("=====================================")
    return result


if __name__ == "__main__":
    # Cho phép truyền đường dẫn ảnh qua argument
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    test_deplot(path)
