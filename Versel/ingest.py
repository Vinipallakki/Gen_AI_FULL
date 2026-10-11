import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma = chromadb.PersistentClient(path="chroma_db")


with open("data/knowledge.txt","r",encoding="utf-8") as f:
    docs=[l.strip() for l in f if l.strip()]

for i,doc in enumerate(docs):
    emb=client.embeddings.create(model="text-embedding-3-small",input=doc).data[0].embedding
    collection.add(ids=[str(i)],documents=[doc],embeddings=[emb])

print("Vector DB Created Successfully")
