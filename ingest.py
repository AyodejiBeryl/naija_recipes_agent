"""
Naija Recipes Agent - PDF Ingestion Script
Parses Nigerian cookbook PDFs, chunks them, and stores embeddings in ChromaDB.
Run this once (or whenever you add new source PDFs).
"""

import os
import sys
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration
SOURCES_DIR = os.path.join(os.path.dirname(__file__), "sources")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "naija_recipes"

# PDF files to ingest
PDF_FILES = [
    {
        "path": os.path.join(SOURCES_DIR, "_OceanofPDF.com_All_Nigerian_Recipes_Cookbook_-_Flo_Madubike.pdf"),
        "source_name": "All Nigerian Recipes Cookbook - Flo Madubike",
    },
    {
        "path": os.path.join(SOURCES_DIR, "520413611-All-Nigerian-Recipes-Cookbook.pdf"),
        "source_name": "All Nigerian Recipes Cookbook",
    },
    {
        "path": os.path.join(SOURCES_DIR, "classicnaijafoodrecipes.pdf"),
        "source_name": "Classic Naija Food Recipes",
    },
]


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_num + 1,
            })
    doc.close()
    return pages


def create_documents(pdf_info: dict, pages: list[dict]) -> list[Document]:
    """Create LangChain Document objects from extracted pages."""
    documents = []
    for page_data in pages:
        doc = Document(
            page_content=page_data["text"],
            metadata={
                "source": pdf_info["source_name"],
                "page": page_data["page"],
                "file": os.path.basename(pdf_info["path"]),
            },
        )
        documents.append(doc)
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def ingest():
    """Main ingestion pipeline."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file.")
        print("Create a .env file with: OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    all_documents = []

    for pdf_info in PDF_FILES:
        pdf_path = pdf_info["path"]
        if not os.path.exists(pdf_path):
            print(f"WARNING: File not found: {pdf_path}")
            print(f"  Skipping {pdf_info['source_name']}")
            continue

        print(f"Processing: {pdf_info['source_name']}")
        print(f"  File: {os.path.basename(pdf_path)}")

        # Extract text
        pages = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(pages)} pages with text")

        # Create documents
        documents = create_documents(pdf_info, pages)
        all_documents.extend(documents)

    if not all_documents:
        print("ERROR: No documents were extracted. Check that PDFs are in the sources/ folder.")
        sys.exit(1)

    # Chunk documents
    print(f"\nTotal pages extracted: {len(all_documents)}")
    chunks = chunk_documents(all_documents)
    print(f"Total chunks created: {len(chunks)}")

    # Create embeddings and store in ChromaDB
    print(f"\nCreating embeddings and storing in ChromaDB...")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Directory: {CHROMA_DIR}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    # Clear existing collection if it exists
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("  Clearing existing collection...")
        import shutil
        shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"\nIngestion complete!")
    print(f"  Chunks stored: {len(chunks)}")
    print(f"  ChromaDB directory: {CHROMA_DIR}")

    # Print summary by source
    source_counts = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "Unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\nChunks per source:")
    for src, count in source_counts.items():
        print(f"  {src}: {count} chunks")


if __name__ == "__main__":
    ingest()
