import ollama
import chromadb
from pathlib import Path
from loguru import logger
from app.core.errors import EmbeddingError, EmptyRetrievalError


# ChromaDB persists to disk at this path.
# Decision: disk persistence over in-memory.
# Reason: in-memory ChromaDB loses all vectors on restart.
# With disk persistence, ingest once, query forever.
# At scale this becomes a managed vector DB (Pinecone, Weaviate) —
# the interface stays identical, only the client changes.
CHROMA_PATH = Path("./chroma_db")
COLLECTION_NAME = "pharma_contracts"
EMBED_MODEL = "nomic-embed-text"   # free, local, no API key needed
TOP_K = 3  # retrieve top 3 chunks per query — enough context, low noise


def get_collection() -> chromadb.Collection:
    """
    Get or create the ChromaDB collection.
    Called on every request — ChromaDB client is lightweight to initialise.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
        # Why cosine? Semantic similarity is directional, not magnitude-based.
        # Two identical sentences with different lengths should score 1.0.
        # Cosine distance captures this. L2 (euclidean) would penalise
        # longer chunks unfairly.
    )
    return collection


def embed_text(text: str) -> list[float]:
    """
    Generate embedding using local Ollama nomic-embed-text model.

    Decision: nomic-embed-text over OpenAI ada-002.
    Reasons:
    1. Zero cost — runs locally via Ollama
    2. GDPR compliance — no data leaves the machine
    3. nomic-embed-text scores comparably to ada-002 on MTEB benchmarks
    Trade-off: slightly slower than API call on first run (model load),
    but subsequent calls are fast once model is in memory.
    """
    try:
        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=text
        )
        return response["embedding"]
    except Exception as e:
        raise EmbeddingError(reason=str(e))


def store_chunks(chunks: list[dict]) -> int:
    """
    Embed and store chunks in ChromaDB.
    Returns count of newly added chunks (skips duplicates by ID).
    """
    collection = get_collection()
    existing_ids = set(collection.get()["ids"])

    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    if not new_chunks:
        logger.info("All chunks already exist in vector store — skipping")
        return 0

    logger.info(f"Embedding {len(new_chunks)} new chunks...")

    embeddings = [embed_text(c["text"]) for c in new_chunks]

    collection.add(
        ids=[c["id"] for c in new_chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in new_chunks],
        metadatas=[c["metadata"] for c in new_chunks]
    )

    logger.info(f"Stored {len(new_chunks)} chunks in ChromaDB")
    return len(new_chunks)


def retrieve_relevant_chunks(query: str) -> list[dict]:
    """
    Embed query and retrieve top-k most relevant contract chunks.

    Returns chunks with their similarity scores so the API layer
    can include confidence signals in its response.
    """
    collection = get_collection()

    if collection.count() == 0:
        raise EmptyRetrievalError(query=query)

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "contract_id": meta.get("contract_id", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "similarity_score": round(1 - dist, 4)
            # Convert cosine distance to similarity:
            # distance=0 means identical → similarity=1.0
            # distance=1 means opposite → similarity=0.0
        })

    logger.info(
        f"Retrieved {len(chunks)} chunks for query. "
        f"Top score: {chunks[0]['similarity_score'] if chunks else 'n/a'}"
    )
    return chunks