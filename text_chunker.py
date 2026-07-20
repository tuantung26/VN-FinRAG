def chunk_text_by_words(text: str, chunk_size: int = 100, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
    
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size - overlap)
        
    return chunks