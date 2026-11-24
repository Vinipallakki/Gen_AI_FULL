
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Simple RAG",
    page_icon="📄",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DB_DIR = BASE_DIR / "chroma_db"

load_dotenv(dotenv_path=ENV_PATH, override=False)

API_KEY = os.getenv("OPENAI_API_KEY")

st.title("📄 Simple RAG using Chroma + OpenAI")
st.caption("Upload a PDF, build a vector database, and ask questions.")


# --------------------------------------------------
# Initialize models
# --------------------------------------------------

def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY
    )


def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=API_KEY
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Configuration")

if API_KEY:
    st.sidebar.success("OpenAI API key loaded from .env")
else:
    st.sidebar.error("OPENAI_API_KEY is missing from .env")

pdf_file = st.sidebar.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

build_button = st.sidebar.button(
    "Build Vector Database",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Build vector database
# --------------------------------------------------

if build_button:

    if not API_KEY:
        st.error(
            "Add OPENAI_API_KEY=your_key to your .env file "
            "and restart the application."
        )
        st.stop()

    if pdf_file is None:
        st.error("Please upload a PDF first.")
        st.stop()

    try:
        with st.spinner("Processing PDF and creating embeddings..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False
            ) as temp_file:
                temp_file.write(pdf_file.getvalue())
                temp_pdf_path = temp_file.name

            try:
                # Load PDF
                loader = PyPDFLoader(temp_pdf_path)
                documents = loader.load()

                if not documents:
                    st.error("No pages were found in the PDF.")
                    st.stop()

                # Split documents into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100
                )

                chunks = splitter.split_documents(documents)

                if not chunks:
                    st.error(
                        "No text chunks found. "
                        "The PDF may be scanned or image-only."
                    )
                    st.stop()

                # Remove previous vector database
                if DB_DIR.exists():
                    shutil.rmtree(DB_DIR)

                # Create embeddings
                embeddings = get_embeddings()

                # Create and persist Chroma database
                db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=str(DB_DIR),
                    collection_name="pdf_documents"
                )

                # Store status for Streamlit reruns
                st.session_state["db_ready"] = True
                st.session_state["pdf_name"] = pdf_file.name
                st.session_state["chunk_count"] = len(chunks)

                st.success("Vector database created successfully!")
                st.write("**PDF:**", pdf_file.name)
                st.write("**Pages:**", len(documents))
                st.write("**Chunks:**", len(chunks))

            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

    except Exception as e:
        st.error(f"Failed to build vector database: {e}")


# --------------------------------------------------
# Database status
# --------------------------------------------------

if st.session_state.get("db_ready") and DB_DIR.exists():
    st.info(
        f"Database ready: {st.session_state.get('pdf_name', 'PDF')} | "
        f"Chunks: {st.session_state.get('chunk_count', 0)}"
    )
elif DB_DIR.exists():
    st.info(
        "An existing vector database was found. "
        "You can ask questions about its stored PDF."
    )


# --------------------------------------------------
# Ask questions
# --------------------------------------------------

st.divider()
st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question about the uploaded PDF",
    placeholder="Example: What is the main topic of this document?"
)

if st.button("Get Answer", type="primary"):

    if not API_KEY:
        st.error("OPENAI_API_KEY is missing from .env.")

    elif not question.strip():
        st.warning("Please enter a question.")

    elif not DB_DIR.exists():
        st.error("Please build the vector database first.")

    else:
        try:
            with st.spinner("Searching the PDF and generating an answer..."):

                # Load existing Chroma database
                embeddings = get_embeddings()

                db = Chroma(
                    persist_directory=str(DB_DIR),
                    embedding_function=embeddings,
                    collection_name="pdf_documents"
                )

                # Retrieve the top 3 relevant chunks
                retriever = db.as_retriever(
                    search_kwargs={"k": 3}
                )

                # Build RAG chain
                qa = RetrievalQA.from_chain_type(
                    llm=get_llm(),
                    retriever=retriever,
                    chain_type="stuff",
                    return_source_documents=True
                )

                # Execute question
                result = qa.invoke({"query": question})

            # Display answer
            st.subheader("✅ Answer")
            st.write(result["result"])

            # Display retrieved chunks
            st.subheader("🔍 Retrieved Chunks")

            source_documents = result.get(
                "source_documents", []
            )

            if not source_documents:
                st.info("No source chunks were returned.")

            for i, doc in enumerate(source_documents, start=1):
                with st.expander(f"Chunk {i}", expanded=True):

                    st.write(doc.page_content)

                    page_number = doc.metadata.get("page")
                    if page_number is not None:
                        st.caption(
                            f"Source page: {page_number + 1}"
                        )

        except Exception as e:
            st.error(f"Question answering failed: {e}")
