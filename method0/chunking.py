def chunkByWords(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """chia nho van ban
    overlap: so tu goi dau giua 2 chunk lien tiep  
    """

    words = text.split()
    chunks = []
    if not words: 
        return chunks

    i=0
    while i < len(words):
        chunk_words = words[i: i+chunk_size]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size - overlap)

    return chunks