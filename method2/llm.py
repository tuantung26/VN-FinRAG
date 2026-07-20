# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import base64
from pprint import pprint

load_dotenv()

FPT_MODEL = os.getenv("FPT_MODEL")
FPT_API_KEY = os.getenv("FPT_API_KEY")
FPT_BASE_URL = os.getenv("FPT_BASE_URL")

llm = ChatOpenAI(
        model=FPT_MODEL,
        api_key=FPT_API_KEY,
        base_url=FPT_BASE_URL,
        temperature=0.2,
        max_tokens=2048,
    )

def get_image_content(path: str):
    with open(path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    message = [
        HumanMessage(content = [
            {"type": "text", "text": "Hãy mô tả ngắn gọn biểu đồ sau bằng cách đưa ra những phần sau một cách đúng cấu trúc: nội dung của biểu đồ bao gồm việc biểu đồ chứa cái gì, cột x, cột y là gì nếu có, Thêm vào đó hãy đưa ra các insight cần có của biểu đồ này (Insight: insight 1, 2....)  (không dùng markdown):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ])
    ]
    response = llm.invoke(message)
    return response.content






