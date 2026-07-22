# ============================================================
# config.py - Cấu hình không bảo mật (non-sensitive settings)
# Các API key vẫn nằm trong .env
# ============================================================

# --- Milvus ---
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "mulRAG"
DIMENSION = 1024

# --- Thư mục lưu trữ ---
IMAGE_DIR = "images"
PAGES_DIR = "pages"
RESULTS_FILE = "result.txt"

# --- FPT Cloud LLM ---
FPT_BASE_URL = "https://mkp-api.fptcloud.com"
FPT_MODEL = "gemma-4-31B-it"

# --- PDF Processing ---
MAX_PDF_TEXT_CHARS = 10000

# --- W&B Inference LLM ---
WANDB_BASE_URL = "https://api.inference.wandb.ai/v1"
WANDB_MODEL = "google/gemma-4-31B-it"
WANDB_PROJECT = "inference/coreweave" 