import tiktoken

def chunk_text(text, chunk_size=200, overlap=50):
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(enc.decode(chunk))

    return chunks
