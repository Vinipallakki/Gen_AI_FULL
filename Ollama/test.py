import ollama

response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "user", "content": "What is AI?"}
    ]
)

print(response['message']['content'])

# ollama pull nomic-embed-text

git config --global user.email "vinayakapallakki@gmail.com"
git config --global user.name "ViniPallakki"