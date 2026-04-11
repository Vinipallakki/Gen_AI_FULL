# ingestion.py

from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    return text