import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
app=FastAPI(title="Simple RAG API")
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db = chromadb.Client()

collection = db.get_or_create_collection("knowledge")

class Question(BaseModel):
    question:str

@app.get("/")
def home():
    return {"message":"Simple RAG API Running"}

@app.post("/chat")
def chat(data:Question):
    qemb=client.embeddings.create(model="text-embedding-3-small",input=data.question).data[0].embedding
    results=collection.query(query_embeddings=[qemb],n_results=3)
    
    context="\n".join(results["documents"][0])
    prompt=f"""You are a helpful assistant.
Answer ONLY using the context.

Context:
{context}

Question:
{data.question}
"""
    resp=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return {"context":context,"answer":resp.choices[0].message.content}
