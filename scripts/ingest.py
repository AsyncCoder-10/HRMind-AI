import os
import sys
from pathlib import Path

# Add the root directory to the python path so we can import src.config
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.config import GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME


def load_documents(data_dir: str):
    documents = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if file.endswith(".pdf"):
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            print(f"Loading DOCX: {file}")
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
    return documents


def main():
    print("Starting ingestion pipeline...")

    # 1. Load Documents
    docs = load_documents("data/")
    if not docs:
        print("No documents found in the 'data/' folder. Add some PDFs/DOCXs first.")
        return

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    # 3. Setup Embeddings & Qdrant
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # 4. Store in Qdrant
    print("Embedding and uploading to Qdrant (this may take a minute)...")
    qdrant = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        prefer_grpc=True,
    )
    print("✅ Ingestion complete! Data is now in Qdrant.")


if __name__ == "__main__":
    main()
