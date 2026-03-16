from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
import uuid
import httpx
from datetime import datetime

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="StreamKar Training Bot",
    description="FAQ knowledge base for StreamKar — supports static FAQs, dynamic FAQs, and auto-learning from user questions",
    version="3.0.0"
)


# ChromaDB Setup
#
# Problem faced: ChromaDB's default in-memory client
# resets every time the server restarts — all FAQs lost.
# Fix: Used PersistentClient so FAQs are saved to disk.
#
# Problem faced: Default embedding model was too slow
# on first load causing timeout errors.
# all-MiniLM-L6-v2 — lightweight and fast.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Collection 1: Admin-added FAQ pairs
collection = chroma_client.get_or_create_collection(
    name="streamkar_faqs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# Collection 2: User questions the bot couldn't answer
#
# Idea: Every time a user asks something the bot doesn't know,
# we auto-save it here. Admin reviews it, adds proper answer
# via /add_faq. This creates a feedback loop —
# bot gets smarter every single day automatically.
#
# Problem faced: Without this, unanswered questions were lost forever.
# We had zero visibility into what real users were actually asking.
# Fix: Separate ChromaDB collection acts as an "inbox" for admin.
unanswered_collection = chroma_client.get_or_create_collection(
    name="streamkar_unanswered",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# ─────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────
class FAQItem(BaseModel):
    question: str
    answer: str
    is_dynamic: bool = False
    api_endpoint: Optional[str] = None
    response_field: Optional[str] = None

class AskRequest(BaseModel):
    question: str
    user_id: Optional[str] = "anonymous"  # track which user asked


# ─────────────────────────────────────────────
# Helper — Fetch Dynamic Answer from Live API
#
# Problem faced: Some StreamKar APIs return nested JSON
# like {"data": {"streamer": {"name": "Priya"}}}
# Simple key lookup fails on nested responses.
# Fix: Used dot notation parsing — response_field = "data.streamer.name"
# ─────────────────────────────────────────────
async def fetch_dynamic_answer(api_endpoint: str, response_field: Optional[str]) -> str:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_endpoint)
            response.raise_for_status()
            data = response.json()

            if response_field:
                keys = response_field.split(".")
                value = data
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return f"Could not extract field '{response_field}' from live data."
                return str(value)

            return str(data)

    except httpx.TimeoutException:
        # Problem faced: Live API times out during peak streaming hours
        # Fix: Return graceful fallback instead of crashing the bot
        return "Live data is temporarily unavailable. Please try again in a moment."

    except httpx.HTTPStatusError as e:
        return f"Could not fetch live data (HTTP {e.response.status_code}). Please try again."

    except Exception:
        return "Unable to fetch live data right now. Please contact StreamKar support."


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "StreamKar Training Bot",
        "status": "running",
        "total_faqs": collection.count(),
        "unanswered_questions": unanswered_collection.count()
    }


# ─────────────────────────────────────────────
# Route 1: Add FAQ (Admin)
# ─────────────────────────────────────────────
@app.post("/add_faq")
async def add_faq(item: FAQItem):
    """
    Admin adds a FAQ to StreamKar's knowledge base.

    Two types supported:

    1. Static FAQ — fixed answer
       { "question": "how do I go live?", "answer": "Tap the Live button" }

    2. Dynamic FAQ — answer fetched from live API at query time
       {
         "question": "who is today's top streamer?",
         "answer": "fetched live",
         "is_dynamic": true,
         "api_endpoint": "https://api.streamkar.com/top-streamer/today",
         "response_field": "streamer_name"
       }

    StreamKar dynamic FAQ use cases:
    - Today's top streamer        → /top-streamer/today
    - Currently live streams      → /streams/live
    - Today's top earner          → /earnings/today
    - Current gift prices         → /gifts/prices
    - Trending streams            → /streams/trending
    - Active events/contests      → /events/active
    """

    if not item.question.strip() or not item.answer.strip():
        raise HTTPException(status_code=400, detail="Question and answer cannot be empty")

    if item.is_dynamic and not item.api_endpoint:
        raise HTTPException(status_code=400, detail="Dynamic FAQs require an api_endpoint")

    # ── Duplicate Check ──────────────────────────────
    # Problem faced: Same FAQ added multiple times caused
    # duplicate results during retrieval.
    # Fix: Reject if cosine distance < 0.1 (near identical match)
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
                detail=f"Duplicate FAQ. Similar question exists: '{duplicate_q}'"
            )

    faq_id = str(uuid.uuid4())
    collection.add(
        documents=[item.question],
        metadatas=[{
            "question": item.question,
            "answer": item.answer,
            "is_dynamic": str(item.is_dynamic),
            "api_endpoint": item.api_endpoint or "",
            "response_field": item.response_field or "",
            "source": "streamkar_admin"
        }],
        ids=[faq_id]
    )

    # ── Verify dynamic API is reachable at add time ──
    # Problem faced: Admins added wrong endpoints that 404'd at query time.
    # Fix: Quick ping at add time to warn admin immediately.
    # ────────────────────────────────────────────────
    api_status = None
    if item.is_dynamic and item.api_endpoint:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(item.api_endpoint)
                api_status = "reachable ✅" if r.status_code == 200 else f"HTTP {r.status_code} ⚠️"
        except Exception:
            api_status = "unreachable ⚠️ — FAQ saved but verify endpoint"

    response = {
        "status": "success",
        "message": "FAQ added to StreamKar knowledge base",
        "id": faq_id,
        "type": "dynamic 🔄" if item.is_dynamic else "static 📌",
        "question": item.question,
        "answer": item.answer if not item.is_dynamic else "fetched live from API",
        "total_faqs": collection.count()
    }

    if api_status:
        response["api_status"] = api_status

    return response


# ─────────────────────────────────────────────
# Route 2: Ask Question (User)
#
# This is the feedback loop endpoint.
#
# Flow:
# 1. User asks a question
# 2. Bot searches ChromaDB for best matching FAQ
# 3a. Match found (confidence > 0.5) → return answer
# 3b. No match → auto-save question to unanswered_collection
#     → admin reviews → adds answer → bot learns
#
# This means every unanswered question makes the bot
# smarter for the next user who asks the same thing.
# ─────────────────────────────────────────────
@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    User asks a question. Bot searches knowledge base and returns best answer.

    If no good answer found:
    - Question is AUTO-SAVED to unanswered questions inbox
    - Admin can review via GET /unanswered
    - Admin adds proper answer via POST /add_faq
    - Next user asking same question gets perfect answer

    This creates a self-improving feedback loop.
    """

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ── No FAQs Yet ──────────────────────────────────
    if collection.count() == 0:
        # Still save the question so admin knows what users need
        _save_unanswered(request.question, request.user_id)
        return {
            "question": request.question,
            "answer": "Our support team is still setting up the knowledge base. We'll get back to you soon!",
            "answered": False,
            "confidence": 0.0
        }

    # ── Search ChromaDB ──────────────────────────────
    results = collection.query(
        query_texts=[request.question],
        n_results=1,
        include=["metadatas", "distances"]
    )

    best_distance = results["distances"][0][0]
    best_meta = results["metadatas"][0][0]

    # Cosine distance → similarity (0 = identical, 1 = completely different)
    confidence = round(1 - best_distance, 2)

    # ── Good Match Found ─────────────────────────────
    if confidence >= 0.5:
        is_dynamic = best_meta.get("is_dynamic") == "True"

        # Static FAQ — return stored answer directly
        if not is_dynamic:
            return {
                "question": request.question,
                "answer": best_meta.get("answer"),
                "matched_faq": best_meta.get("question"),
                "answered": True,
                "confidence": confidence,
                "type": "static 📌"
            }

        # Dynamic FAQ — call live StreamKar API for fresh answer
        live_answer = await fetch_dynamic_answer(
            api_endpoint=best_meta.get("api_endpoint"),
            response_field=best_meta.get("response_field") or None
        )
        return {
            "question": request.question,
            "answer": live_answer,
            "matched_faq": best_meta.get("question"),
            "answered": True,
            "confidence": confidence,
            "type": "dynamic 🔄 (live data)"
        }

    # ── No Good Match — Auto Save to Unanswered ──────
    #
    # Problem faced: Low confidence answers were being shown to users
    # causing confusion and wrong information.
    # Fix: Threshold at 0.5 — below that, admit we don't know
    # and save question for admin to review.
    # ────────────────────────────────────────────────
    _save_unanswered(request.question, request.user_id)

    return {
        "question": request.question,
        "answer": "I don't have a confident answer for this yet. Your question has been saved and our team will add it to the knowledge base soon!",
        "answered": False,
        "confidence": confidence,
        "note": "Question auto-saved for admin review 📥"
    }


# ─────────────────────────────────────────────
# Helper — Save Unanswered Question
# ─────────────────────────────────────────────
def _save_unanswered(question: str, user_id: str = "anonymous"):
    """
    Save unanswered user question to separate ChromaDB collection.
    Checks for duplicates so same question isn't saved 100 times.
    """
    # Don't save duplicate unanswered questions
    if unanswered_collection.count() > 0:
        existing = unanswered_collection.query(
            query_texts=[question],
            n_results=1,
            include=["distances"]
        )
        if existing["distances"][0] and existing["distances"][0][0] < 0.1:
            return  # Already saved this question before

    unanswered_collection.add(
        documents=[question],
        metadatas=[{
            "question": question,
            "user_id": user_id,
            "asked_at": datetime.now().isoformat(),
            "status": "pending_review"
        }],
        ids=[str(uuid.uuid4())]
    )


# ─────────────────────────────────────────────
# Route 3: View Unanswered Questions (Admin)
# ─────────────────────────────────────────────
@app.get("/unanswered")
def get_unanswered():
    """
    Admin reviews all questions the bot couldn't answer.
    Use this to identify gaps in the knowledge base.
    Then add proper answers via POST /add_faq.
    """
    if unanswered_collection.count() == 0:
        return {
            "total": 0,
            "message": "No unanswered questions — bot is handling everything! 🎉",
            "questions": []
        }

    results = unanswered_collection.get(include=["metadatas"])

    questions = [
        {
            "id": id_,
            "question": meta.get("question"),
            "asked_by": meta.get("user_id"),
            "asked_at": meta.get("asked_at"),
            "status": meta.get("status")
        }
        for id_, meta in zip(results["ids"], results["metadatas"])
    ]

    return {
        "total": len(questions),
        "message": f"{len(questions)} question(s) need answers. Add them via POST /add_faq",
        "questions": questions
    }