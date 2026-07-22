# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import openai
import os
import base64
from pprint import pprint
from config import FPT_MODEL, FPT_BASE_URL, WANDB_BASE_URL, WANDB_MODEL, WANDB_PROJECT

load_dotenv()

FPT_API_KEY = os.getenv("FPT_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def get_llm():
    llm = ChatOpenAI(
            model=FPT_MODEL,
            api_key=FPT_API_KEY,
            base_url=FPT_BASE_URL,
            temperature=0.2,
            max_tokens=2048,
        )
    return llm


def get_llm_wandb():
    client = openai.OpenAI(
        base_url=WANDB_BASE_URL,
        api_key=WANDB_API_KEY,
    )
    return client





