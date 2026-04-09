from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load API Key
# -----------------------------
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file")

client = OpenAI(api_key=api_key)

# -----------------------------
# Sample Text
# -----------------------------
text = """Artificial Intelligence is transforming industries.
Machine learning is a subset of AI.
Deep learning uses neural networks.
Python is widely used for AI development.
It has many libraries like TensorFlow and PyTorch."""

# -----------------------------
# Step 1: Split into sentences
# -----------------------------
sentences = [s.strip() for s in text.split("\n") if s.strip()]

# -----------------------------
# Step 2: Get embeddings (batch call)
# -----------------------------
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=sentences
)

embeddings = [item.embedding for item in response.data]

# -----------------------------
# Step 3: Semantic Chunking
# -----------------------------
chunks = []
current_chunk = [sentences[0]]

for i in range(1, len(sentences)):
    sim = cosine_similarity(
        [embeddings[i-1]],
        [embeddings[i]]
    )[0][0]
    print(sim)

    if sim > 0.7:
        current_chunk.append(sentences[i])
    else:
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i]]

# Add last chunk
chunks.append(" ".join(current_chunk))

# -----------------------------
# Output
# -----------------------------
print("\n--- Semantic Chunks ---\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("-" * 40)