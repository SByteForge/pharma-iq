import time
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import ollama

from app.ingestion.pdf_chunker import process_contract_pdf
from app.ingestion.vector_store import store_chunks, retrieve_relevant_chunks
from app.core.errors import OllamaUnavailableError, EmptyRetrievalError
from app.session.store import SessionStore


# --- Lifespan: runs on startup and shutdown ---
# Why lifespan over @app.on_event?
# on_event is deprecated in FastAPI 0.93+. Lifespan is the current
# pattern — it uses a context manager so startup and shutdown logic
# live together, making the dependency clear.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PharmaIQ starting — verifying Ollama connection...")
    try:
        ollama.list()
        logger.info("Ollama connection confirmed")
    except Exception:
        logger.error("Ollama not reachable — start with: ollama serve")
    yield
    logger.info("PharmaIQ shutting down")


app = FastAPI(
    title="PharmaIQ",
    description="Privacy-first pharma contract intelligence. Runs 100% locally.",
    version="0.1.0",
    lifespan=lifespan
)

session_store = SessionStore()

from app.core.config import OLLAMA_MODEL


# --- Request / Response models ---

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language question about pharma contracts"
    )
    session_id: str = Field(
        default="default",
        description="Session ID for conversation continuity"
    )

class ChunkReference(BaseModel):
    source: str
    contract_id: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[ChunkReference]
    latency_ms: float


# --- Endpoints ---

@app.get("/health")
def health_check():
    """Verify API and Ollama are reachable."""
    try:
        ollama.list()
        ollama_status = "ok"
    except Exception:
        ollama_status = "unreachable"

    return {
        "status": "ok",
        "ollama": ollama_status,
        "version": app.version
    }


@app.post("/ingest")
async def ingest_contract(file: UploadFile = File(...)):
    """
    Upload a pharma contract PDF and index it for querying.

    Decision: UploadFile over accepting a file path.
    Reason: file path approach requires the client to have access to
    the server filesystem — not viable in any real deployment.
    UploadFile streams the file through the HTTP request, works locally
    and in any cloud deployment without change.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail="Only PDF files are supported. Convert other formats to PDF first."
        )

    start = time.time()

    # Write upload to temp file — pypdf needs a file path, not a stream
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        chunks = process_contract_pdf(tmp_path)
        # Override source metadata with original filename
        for chunk in chunks:
            chunk["metadata"]["source"] = file.filename
            chunk["metadata"]["contract_id"] = Path(file.filename).stem

        stored_count = store_chunks(chunks)
        latency = round((time.time() - start) * 1000, 2)

        logger.info(
            f"Ingested '{file.filename}': "
            f"{len(chunks)} chunks, {stored_count} new, {latency}ms"
        )

        return {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "new_chunks_stored": stored_count,
            "message": "ready to query" if stored_count > 0 else "already indexed",
            "latency_ms": latency
        }
    finally:
        tmp_path.unlink(missing_ok=True)  # always clean up temp file


@app.post("/query", response_model=QueryResponse)
def query_contracts(request: QueryRequest):
    """
    Ask a natural language question about ingested pharma contracts.

    Architecture decision — RAG over fine-tuning:
    Contract data changes frequently (new tiers, amended clauses).
    Fine-tuning requires retraining on every change — expensive and slow.
    RAG updates the knowledge base by re-ingesting the new document.
    No retraining. No downtime. Cost is zero beyond storage.
    """
    start = time.time()

    # Step 1: retrieve relevant contract chunks
    try:
        chunks = retrieve_relevant_chunks(request.question)
    except EmptyRetrievalError:
        raise

    # Step 2: build context from retrieved chunks
    context = "\n\n---\n\n".join([c["text"] for c in chunks])

    # Step 3: get conversation history for this session
    history = session_store.get_history(request.session_id)

    # Step 4: build the prompt
    # System prompt is explicit about grounding — model must not answer
    # from training knowledge, only from the provided contract context.
    # This is critical for pharma: hallucinated contract terms are
    # a compliance risk, not just a quality issue.
    system_prompt = """You are a pharma contract analyst assistant.
Answer questions using ONLY the contract excerpts provided below.
If the answer is not in the excerpts, say exactly:
'This information is not found in the ingested contracts.'
Never guess or use general knowledge about pharma contracts.
Always cite which contract your answer comes from.

CONTRACT EXCERPTS:
{context}""".format(context=context)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": request.question})

    # Step 5: call local Ollama
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        answer = response["message"]["content"]
    except Exception as e:
        raise OllamaUnavailableError()

    # Step 6: persist conversation turn
    session_store.add_turn(
        session_id=request.session_id,
        user_message=request.question,
        assistant_message=answer
    )

    latency = round((time.time() - start) * 1000, 2)
    logger.info(
        f"Query answered in {latency}ms | "
        f"session={request.session_id} | "
        f"top_score={chunks[0]['similarity_score']}"
    )

    return QueryResponse(
        answer=answer,
        session_id=request.session_id,
        sources=[
            ChunkReference(
                source=c["source"],
                contract_id=c["contract_id"],
                similarity_score=c["similarity_score"]
            ) for c in chunks
        ],
        latency_ms=latency
    )