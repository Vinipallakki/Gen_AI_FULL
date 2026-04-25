import weaviate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ==============================
# 🔐 CONFIG (CHANGE THESE)
# ==============================

WEAVIATE_URL = "enter your weaviate url here"  # e.g. "https://your-cluster.weaviate.network"
WEAVIATE_API_KEY = "Enter your weaviate API key here"  # e.g. "sk-xxxxxx"

OPENAI_API_KEY = "enter your OpenAI API key here"  # e.g. "sk-xxxxxx"

PDF_PATH = "Enter the path to your PDF here"  # e.g. "path/to/your/document.pdf"

INDEX_NAME = "HRDocs"

# ==============================
# 📄 LOAD PDF
# ==============================

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# ==============================
# ✂️ SPLIT INTO CHUNKS
# ==============================

print("Splitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# ==============================
# 🔗 CONNECT TO WEAVIATE
# ==============================

print("Connecting to Weaviate...")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
)

# ==============================
# 🗄️ VECTOR STORE
# ==============================

print("Creating vector store...")

embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

vectorstore = WeaviateVectorStore(
    client=client,
    index_name=INDEX_NAME,
    text_key="text",
    embedding=embedding
)

# ==============================
# 📥 STORE DATA (only once)
# ==============================

print("Uploading documents to Weaviate...")
vectorstore.add_documents(chunks)

print("✅ Documents stored successfully!")

# ==============================
# 🤖 RAG QA CHAIN (using LCEL)
# ==============================

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

retriever = vectorstore.as_retriever()

# Create a simple prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question based on the provided context.
Context: {context}
Question: {question}
Answer:"""
)

# Build the LCEL chain
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

qa = (
    RunnablePassthrough.assign(
        context=RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs)
    )
    | prompt
    | llm
    | StrOutputParser()
)

# ==============================
# 💬 ASK QUESTIONS LOOP
# ==============================

print("\n🎉 RAG system ready! Ask questions (type 'exit' to quit)\n")

while True:
    query = input("❓ Your question: ")

    if query.lower() == "exit":
        break

    answer = qa.invoke({"question": query})

    print("\n💡 Answer:", answer)
    print("-" * 50)