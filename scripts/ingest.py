import os
import sys
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Add the root directory to the python path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # New local embedding
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME


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

    docs = load_documents("data/")
    if not docs:
        print("No documents found in the 'data/' folder.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    # 1. Use a local, open-source embedding model (No API limits)
    print("Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️ Deleted old collection '{COLLECTION_NAME}' to start fresh.")
    except Exception:
        pass

    # 2. Create collection with the correct dimension for MiniLM (384)
    print("Creating new collection in Qdrant...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 3. Process in larger batches since there are no API rate limits
    print("Embedding and uploading to Qdrant. This will be fast!")
    batch_size = 100
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num} of {total_batches}...")
        vectorstore.add_documents(batch)

    print("✅ Ingestion complete! Data is now in Qdrant.")


if __name__ == "__main__":
    main()
