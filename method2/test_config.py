"""
Test toàn diện: kiểm tra tất cả modules, 2 LLM (FPT + W&B), và imports không conflict.
"""
import sys

print("=" * 55)
print("  FULL INTEGRATION TEST - Check for conflicts")
print("=" * 55)

errors = []
passed = 0

# --- 1. Config ---
print("\n[1] config.py imports...")
try:
    from config import (
        MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, DIMENSION,
        IMAGE_DIR, PAGES_DIR, RESULTS_FILE,
        FPT_BASE_URL, FPT_MODEL, MAX_PDF_TEXT_CHARS,
        WANDB_BASE_URL, WANDB_MODEL,
    )
    print(f"    FPT:  model={FPT_MODEL}, base_url={FPT_BASE_URL}")
    print(f"    W&B:  model={WANDB_MODEL}, base_url={WANDB_BASE_URL}")
    print("    [OK]")
    passed += 1
except Exception as e:
    print(f"    [FAIL] {e}")
    errors.append(("config.py", str(e)))

# --- 2. .env API keys ---
print("\n[2] .env API keys...")
try:
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    keys = {
        "JINA_API_KEY": os.getenv("JINA_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "FPT_API_KEY": os.getenv("FPT_API_KEY"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
    }
    for name, val in keys.items():
        status = f"***{val[-6:]}" if val else "MISSING!"
        print(f"    {name}: {status}")
    
    missing = [k for k, v in keys.items() if not v]
    if missing:
        raise ValueError(f"Missing: {missing}")
    print("    [OK]")
    passed += 1
except Exception as e:
    print(f"    [FAIL] {e}")
    errors.append((".env keys", str(e)))

# --- 3. Module imports (no heavy models) ---
print("\n[3] Module imports...")
modules_ok = 0
modules_total = 0

for mod_name, import_stmt in [
    ("text_chunker", "from text_chunker import chunk_text_by_words"),
    ("SplitPDF", "from SplitPDF import SplitPDF"),
    ("embedding", "from embedding import Jina"),
    ("milvusdb", "from milvusdb import get_collection"),
    ("OCR", "from OCR import OCR"),
]:
    modules_total += 1
    try:
        exec(import_stmt)
        print(f"    {mod_name}: [OK]")
        modules_ok += 1
    except Exception as e:
        print(f"    {mod_name}: [FAIL] {e}")
        errors.append((mod_name, str(e)))

if modules_ok == modules_total:
    passed += 1

# --- 4. llm.py - both LLMs ---
print("\n[4] llm.py - check both LLM functions exist...")
try:
    from llm import get_llm, get_llm_wandb, get_image_content
    
    # FPT LLM
    fpt_llm = get_llm()
    print(f"    get_llm(): [OK] type={type(fpt_llm).__name__}")
    
    # W&B LLM
    wandb_client = get_llm_wandb()
    print(f"    get_llm_wandb(): [OK] type={type(wandb_client).__name__}")
    
    # Check they are different objects
    assert type(fpt_llm) != type(wandb_client), "Both LLMs should be different types!"
    print("    No conflict: FPT (ChatOpenAI) vs W&B (openai.OpenAI) [OK]")
    
    print("    [OK]")
    passed += 1
except Exception as e:
    print(f"    [FAIL] {e}")
    errors.append(("llm.py", str(e)))

# --- 5. Wildcard import conflict check ---
print("\n[5] Wildcard import check (from llm import *)...")
try:
    # Simulate what RAG.py and imageProcess.py do
    from llm import *
    
    # Check critical names are accessible
    assert callable(get_llm), "get_llm not callable"
    assert callable(get_llm_wandb), "get_llm_wandb not callable"
    assert callable(get_image_content), "get_image_content not callable"
    print("    get_llm, get_llm_wandb, get_image_content: all accessible [OK]")
    passed += 1
except Exception as e:
    print(f"    [FAIL] {e}")
    errors.append(("wildcard import", str(e)))

# --- 6. W&B LLM actual API call ---
print("\n[6] W&B LLM API call test...")
try:
    from llm import get_llm_wandb
    from config import WANDB_MODEL
    
    client = get_llm_wandb()
    r = client.chat.completions.create(
        model=WANDB_MODEL,
        messages=[{"role": "user", "content": "Reply with just the word OK"}],
        max_tokens=10,
    )
    response_text = r.choices[0].message.content.strip()
    print(f"    Response: '{response_text}'")
    print("    [OK]")
    passed += 1
except Exception as e:
    print(f"    [FAIL] {e}")
    errors.append(("W&B API call", str(e)))

# --- Summary ---
total = passed + len(errors)
print("\n" + "=" * 55)
if not errors:
    print(f"  ALL TESTS PASSED ({passed}/{passed})")
else:
    print(f"  {passed}/{total} passed, {len(errors)} failed:")
    for name, err in errors:
        print(f"    - {name}: {err}")
print("=" * 55)
