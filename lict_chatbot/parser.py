import os
import json
import streamlit as st
from pypdf import PdfReader

# Attempt to import LlamaParse
try:
    from llama_parse import LlamaParse
    HAS_LLAMA_PARSE = True
except ImportError:
    HAS_LLAMA_PARSE = False

def parse_txt_or_md(file_path: str) -> str:
    """Parses plain text or markdown files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_json(file_path: str) -> str:
    """Parses JSON files and returns a structured string format."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    # Return formatted string representation of the JSON
    return json.dumps(data, indent=2)

def parse_pdf_pypdf(file_path: str) -> str:
    """Fallback PDF parsing using standard PyPDF library."""
    reader = PdfReader(file_path)
    text_content = []
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_content.append(text)
    return "\n\n".join(text_content)

def parse_pdf_llama_parse(file_path: str, api_key: str) -> str:
    """Parses PDF using LlamaParse API."""
    if not HAS_LLAMA_PARSE:
        raise ImportError("llama-parse library is not installed.")
        
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",  # Output markdown structure
        verbose=True
    )
    
    # load_data is blocking but runs sync. It uses nest_asyncio internally if needed.
    documents = parser.load_data(file_path)
    text_content = [doc.text for doc in documents if doc.text]
    return "\n\n".join(text_content)

def parse_document(file_path: str, file_extension: str) -> tuple[str, str]:
    """
    Parses a document based on its extension.
    Returns:
        (parsed_text, parser_used)
    """
    ext = file_extension.lower()
    
    if ext in [".txt", ".md"]:
        return parse_txt_or_md(file_path), "Standard Text Reader"
    elif ext == ".json":
        return parse_json(file_path), "JSON Formatter"
    elif ext == ".pdf":
        # Check for LlamaParse key
        llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
        # Streamlit secrets support
        if not llama_key:
            try:
                if "LLAMA_CLOUD_API_KEY" in st.secrets:
                    llama_key = st.secrets["LLAMA_CLOUD_API_KEY"]
            except Exception:
                pass
            
        if HAS_LLAMA_PARSE and llama_key and llama_key.startswith("llx-"):
            try:
                # Apply nest_asyncio just in case streamlit's thread needs it
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except Exception:
                    pass
                return parse_pdf_llama_parse(file_path, llama_key), "LlamaParse"
            except Exception as e:
                # Fallback on LlamaParse API error
                st.warning(f"LlamaParse error: {e}. Falling back to standard PDF reader.")
                return parse_pdf_pypdf(file_path), "PyPDF Fallback (LlamaParse failed)"
        else:
            if not llama_key:
                st.info("LLAMA_CLOUD_API_KEY not configured. Using standard PyPDF parser.")
            return parse_pdf_pypdf(file_path), "PyPDF"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
