from ingestion import load_pdf
from preprocessing import clean_text
from chunking import chunk_text
from vector_store import create_collection, insert_chunks
from retriever import retrieve
from rag_pipeline import generate_answer

from ingestion import load_pdf

text = load_pdf("data/hr_policy.pdf")

# text = load_documents("data/sample.txt")
cleaned = clean_text(text)
chunks = chunk_text(cleaned)

create_collection()
insert_chunks(chunks)

query = "What is AI?"
contexts = retrieve(query)
answer = generate_answer(query, contexts)

print(answer)
