import ollama

text = "AI is transforming industries"

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=text
)

embedding = response["embedding"]

print("Embedding (first 5 values):", embedding[:5])
print("Vector length:", len(embedding))

# import ollama

# texts = [
#     "AI is powerful",
#     "Machine learning is part of AI",
#     "Football is a sport"
# ]

# embeddings = []

# for t in texts:
#     res = ollama.embeddings(
#         model="nomic-embed-text",
#         prompt=t
#     )
#     embeddings.append(res["embedding"])

# print("Total embeddings:", len(embeddings))
# print("One vector length:", len(embeddings[0]))