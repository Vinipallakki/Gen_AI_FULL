import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_PATH = Path("data/hr_policy.pdf")
CHROMA_DIR = "chroma_db"

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            "Put your single HR PDF at data/hr_policy.pdf"
        )

    print(f"Loading: {PDF_PATH}")
    documents = PyPDFLoader(str(PDF_PATH)).load()
    print(f"Pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("HR PDF indexed successfully.")

if __name__ == "__main__":
    main()
