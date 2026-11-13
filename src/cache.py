import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, CACHE_COLLECTION_NAME

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY
)

# Initialize cache collection if it doesn't exist
collections = [c.name for c in client.get_collections().collections]
if CACHE_COLLECTION_NAME not in collections:
    client.create_collection(
        collection_name=CACHE_COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )


def check_cache(query: str):
    query_embedding = embeddings.embed_query(query)
    search_result = client.query_points(
        collection_name=CACHE_COLLECTION_NAME,
        query=query_embedding,
        limit=1,
        score_threshold=0.95,  # 95% semantic similarity
    ).points

    if search_result:
        return search_result[0].payload["response"]
    return None


def update_cache(query: str, response: str):
    query_embedding = embeddings.embed_query(query)
    client.upsert(
        collection_name=CACHE_COLLECTION_NAME,
        points=[
            PointStruct(
                id=hashlib.md5(query.encode()).hexdigest(),
                vector=query_embedding,
                payload={"query": query, "response": response},
            )
        ],
    )
