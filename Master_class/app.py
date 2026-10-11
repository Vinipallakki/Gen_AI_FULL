import os
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


st.set_page_config(page_title="Simple RAG", page_icon="📄")

st.title("📄 Simple RAG using Chroma + OpenAI")


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

pdf_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

build = st.sidebar.button("Build Vector Database")

# -----------------------------
# Build Database
# -----------------------------

if build:

    if api_key == "":
        st.error("Enter API Key")
        st.stop()

    if pdf_file is None:
        st.error("Upload PDF")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    loader = PyPDFLoader("temp.pdf")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db"
    )

    st.success(f"Database Created Successfully!")

    st.write("Chunks:", len(chunks))

# -----------------------------
# Ask Question
# -----------------------------

question = st.text_input("Ask a Question")

if st.button("Get Answer"):

    if api_key == "":
        st.error("Enter API Key")
        st.stop()

    if not os.path.exists("chroma_db"):
        st.error("Please Build Database First")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
        search_kwargs={
            "k":3
        }
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    result = qa.invoke(question)

    st.subheader("Answer")

    st.write(result["result"])

    st.subheader("Retrieved Chunks")

    for i, doc in enumerate(result["source_documents"], 1):

        st.markdown(f"### Chunk {i}")

        st.write(doc.page_content)

        st.write("---")