import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import QDRANT_URL, QDRANT_API_KEY, CACHE_COLLECTION_NAME

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Use the same local embedding model as the ingestion script
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

# Initialize cache collection if it doesn't exist
collections = [c.name for c in client.get_collections().collections]
if CACHE_COLLECTION_NAME not in collections:
    print(f"Creating new cache collection: {CACHE_COLLECTION_NAME}")
    client.create_collection(
        collection_name=CACHE_COLLECTION_NAME,
        # The local MiniLM model outputs 384-dimensional vectors
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def check_cache(query: str):
    """Check if a highly similar query exists in the cache."""
    query_embedding = embeddings.embed_query(query)

    search_result = client.query_points(
        collection_name=CACHE_COLLECTION_NAME,
        query=query_embedding,
        limit=1,
        score_threshold=0.95,  # 95% semantic similarity threshold
    ).points

    if search_result:
        return search_result[0].payload["response"]
    return None


def update_cache(query: str, response: str):
    """Save a new query and its response to the cache."""
    query_embedding = embeddings.embed_query(query)

    client.upsert(
        collection_name=CACHE_COLLECTION_NAME,
        points=[
            PointStruct(
                id=hashlib.md5(
                    query.encode()
                ).hexdigest(),  # Unique ID based on the query text
                vector=query_embedding,
                payload={"query": query, "response": response},
            )
        ],
    )
