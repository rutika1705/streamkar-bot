from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import uuid

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="StreamKar Training Bot",
    description="FAQ knowledge base for StreamKar support agents and users",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# ChromaDB Setup
#
# Problem faced: ChromaDB's default in-memory client
# resets every time the server restarts — all FAQs lost.
#
# Fix: Used PersistentClient so FAQs are saved to disk
# and survive server restarts.
#
# Problem faced: Default embedding model was too slow
# on first load causing timeout errors.
#
# Fix: Pinned to all-MiniLM-L6-v2 — lightweight, fast,
# and good enough for FAQ semantic matching.
# ─────────────────────────────────────────────
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="streamkar_faqs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # cosine similarity for better text matching
)

# ─────────────────────────────────────────────
# Request Model
# ─────────────────────────────────────────────
class FAQItem(BaseModel):
    question: str
    answer: str

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "StreamKar Training Bot",
        "status": "running",
        "total_faqs_stored": collection.count()
    }


@app.post("/add_faq")
def add_faq(item: FAQItem):
    """
    Adds a FAQ to StreamKar's knowledge base.

    - Embeds the question using SentenceTransformers
    - Stores in ChromaDB with persistent storage
    - Checks for duplicate questions before adding

    Example:
        POST /add_faq
        { "question": "how do I go live?", "answer": "Tap the Live button on home screen" }
    """

    # Validate input
    if not item.question.strip() or not item.answer.strip():
        raise HTTPException(
            status_code=400,
            detail="Question and answer cannot be empty"
        )

    # ── Duplicate Check ──────────────────────────────
    # Problem faced: Same FAQ was being added multiple
    # times causing duplicate results during retrieval.
    #
    # Fix: Before adding, query ChromaDB for similar
    # questions. If cosine distance < 0.1 (very close
    # match), reject as duplicate.
    # ────────────────────────────────────────────────
    if collection.count() > 0:
        existing = collection.query(
            query_texts=[item.question],
            n_results=1,
            include=["distances", "metadatas"]
        )
        if existing["distances"][0] and existing["distances"][0][0] < 0.1:
            duplicate_q = existing["metadatas"][0][0].get("question")
            raise HTTPException(
                status_code=409,
                detail=f"Duplicate FAQ detected. Similar question already exists: '{duplicate_q}'"
            )

    # Add to ChromaDB
    faq_id = str(uuid.uuid4())
    collection.add(
        documents=[item.question],        # question is embedded for semantic search
        metadatas=[{
            "question": item.question,
            "answer": item.answer,
            "source": "streamkar_admin"
        }],
        ids=[faq_id]
    )

    return {
        "status": "success",
        "message": "FAQ added to StreamKar knowledge base",
        "id": faq_id,
        "question": item.question,
        "answer": item.answer,
        "total_faqs": collection.count()
    }
