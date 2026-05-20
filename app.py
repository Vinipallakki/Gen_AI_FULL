from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Create vector DB
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Create prompt template
template = """Answer the following question based on the provided context:

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

@app.get("/")
def home():
    return {"message": "LangChain PDF Bot Running"}

@app.get("/ask")
def ask(query: str):
    response = qa_chain.invoke(query)
    return {"answer": response.content}
