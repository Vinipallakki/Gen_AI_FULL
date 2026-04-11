import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION = "rag_collection"

def create_collection():
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def insert_chunks(chunks):
    points = []
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        points.append(PointStruct(id=i, vector=vector, payload={"text": chunk}))
    qdrant.upsert(collection_name=COLLECTION, points=points)
