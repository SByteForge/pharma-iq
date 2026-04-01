import hashlib
from pathlib import Path
from loguru import logger
from pypdf import PdfReader
from app.core.errors import DocumentParseError


# --- Why 512 tokens with 50 overlap? ---
# Pharma contracts have clause-level logic. A rebate tier clause
# is typically 200-400 words. 512 tokens captures a full clause
# plus neighbouring context. Overlap of 50 tokens ensures a clause
# split at a chunk boundary still appears in full in one of the chunks.
# Too large (1024+): retrieval becomes imprecise, unrelated clauses mixed.
# Too small (128): multi-condition clauses get split, losing logical context.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 50  # discard chunks that are just whitespace or headers


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract raw text from a PDF file.

    Decision: pypdf over pdfplumber or pymupdf.
    Reason: standard pharma contracts are digital PDFs, not scanned.
    pypdf handles these cleanly with zero native dependencies.
    If we later need scanned contract support, we swap to pymupdf + tesseract
    without changing any other part of the pipeline.
    """
    try:
        reader = PdfReader(str(file_path))
        pages = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {page_num + 1}]\n{text.strip()}")

        if not pages:
            raise DocumentParseError(
                filename=file_path.name,
                reason="PDF contains no extractable text. May be scanned or image-based."
            )

        full_text = "\n\n".join(pages)
        logger.info(f"Extracted {len(pages)} pages from {file_path.name}")
        return full_text

    except DocumentParseError:
        raise
    except Exception as e:
        raise DocumentParseError(filename=file_path.name, reason=str(e))


def chunk_text(text: str, source_filename: str) -> list[dict]:
    """
    Split text into overlapping chunks with metadata.

    Returns a list of dicts — each chunk carries metadata alongside content.
    Metadata is critical for filtered retrieval later:
    - Filter by contract_id to answer questions about a specific contract
    - Filter by source to cite which document an answer came from
    - chunk_index helps reconstruct the original document order if needed

    Decision: manual chunking over LangChain's text splitters.
    Reason: LangChain splitters work on character count, not token count.
    At 512 chars we'd get half the context we expect. We split on words
    here as a practical approximation — good enough for V1, documented
    upgrade path to tiktoken-based splitting for V2.
    """
    words = text.split()
    chunks = []
    chunk_index = 0

    i = 0
    while i < len(words):
        chunk_words = words[i: i + CHUNK_SIZE]
        chunk_text_str = " ".join(chunk_words)

        # Skip chunks that are too short to carry meaningful content
        if len(chunk_text_str.strip()) >= MIN_CHUNK_LENGTH:
            # Deterministic chunk ID based on content hash
            # Why: if the same document is re-ingested, we can detect
            # and skip duplicate chunks without scanning the full index
            chunk_id = hashlib.md5(chunk_text_str.encode()).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": chunk_text_str,
                "metadata": {
                    "source": source_filename,
                    "contract_id": Path(source_filename).stem,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_words),
                }
            })
            chunk_index += 1

        # Move forward by CHUNK_SIZE minus overlap
        i += CHUNK_SIZE - CHUNK_OVERLAP

    logger.info(
        f"Chunked '{source_filename}' into {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def process_contract_pdf(file_path: Path) -> list[dict]:
    """
    Full ingestion pipeline for a single contract PDF.
    Entry point called by the API layer.
    """
    logger.info(f"Processing contract: {file_path.name}")
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text, file_path.name)
    return chunks