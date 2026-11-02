from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------------------------------
# Step 1: Documents
# ----------------------------------------------------
documents = [
    "Pinecone is a vector database.",
    "FAISS is an open source vector database.",
    "BM25 is a keyword search algorithm.",
    "Hybrid Search combines BM25 and Vector Search."
]

# ----------------------------------------------------
# Step 2: Build BM25 Index
# ----------------------------------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
# "Pinecone" "is" "a" "vector database".
bm25 = BM25Okapi(tokenized_docs)


# ----------------------------------------------------
# Step 3: Build Vector Embeddings
# ----------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") #DIMENTION 384
doc_embeddings = embedding_model.encode(documents)


# ----------------------------------------------------
# Step 4: User Query
# ----------------------------------------------------
query = "who is Virat kohli"
[384 token embedding] = embedding_model.encode([query])

# ----------------------------------------------------
# Step 5: BM25 Search
# ----------------------------------------------------
bm25_scores = bm25.get_scores(query.lower().split())

print("\n========== BM25 Scores ==========")
for doc, score in zip(documents, bm25_scores):
    print(f"{score:.3f} --> {doc}")

# ----------------------------------------------------
# Step 6: Vector Search
# ----------------------------------------------------
query_embedding = embedding_model.encode(query)

vector_scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

print("\n========== Vector Scores ==========")
for doc, score in zip(documents, vector_scores):
    print(f"{score:.3f} --> {doc}")

# ----------------------------------------------------
# Step 7: Normalize BM25
# ----------------------------------------------------
bm25_normalized = (
    bm25_scores - np.min(bm25_scores)
) / (
    np.max(bm25_scores) - np.min(bm25_scores)
    + 1e-10
)

print("\n========== Normalized BM25 ==========")
for doc, score in zip(documents, bm25_normalized):
    print(f"{score:.3f} --> {doc}")

# ----------------------------------------------------
# Step 8: Hybrid Search
# ----------------------------------------------------
alpha = 0.7

hybrid_scores = (
    alpha * vector_scores
    +
    (1-alpha) * bm25_normalized
)

print("\n========== Hybrid Scores ==========")
for doc, score in zip(documents, hybrid_scores):
    print(f"{score:.3f} --> {doc}")

# ----------------------------------------------------
# Step 9: Hybrid Ranking
# ----------------------------------------------------
results = sorted(
    zip(documents, hybrid_scores),
    key=lambda x: x[1],
    reverse=True
)

print("\n========== Hybrid Ranking ==========")
for rank, (doc, score) in enumerate(results, start=1):
    print(f"{rank}. {score:.3f} --> {doc}")

# ====================================================
# STEP 10 : RE-RANKING
# ====================================================

print("\nLoading Re-Ranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Take Top-3 Hybrid Results
top_docs = [doc for doc, _ in results[:3]]

# Query-Document Pairs
pairs = [
    (query, doc)
    for doc in top_docs
]

# Predict relevance scores
rerank_scores = reranker.predict(pairs)

print("\n========== Re-Ranker Scores ==========")
for doc, score in zip(top_docs, rerank_scores):
    print(f"{score:.3f} --> {doc}")

# Sort using reranker scores
final_results = sorted(
    zip(top_docs, rerank_scores),
    key=lambda x: x[1],
    reverse=True
)

print("\n========== Final Ranking After Re-Ranking ==========")

for rank, (doc, score) in enumerate(final_results, start=1):
    print(f"{rank}. {score:.3f} --> {doc}")