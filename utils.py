import fitz  # PyMuPDF

def pdf_to_text(pdf_path):
    """Convert PDF to text."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        if not text.strip():  # Check if text is empty after extraction
            print(f"No text found in the PDF: {pdf_path}")
            return None
        return text.strip()
    except Exception as e:
        print(f"An error occurred while reading {pdf_path}: {e}")
        return None
