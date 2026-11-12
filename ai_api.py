"""
FastAPI endpoints for the existing Gemini + LangGraph assistant.
Run: uvicorn ai_api:app --reload --host 127.0.0.1 --port 8001
"""
from collections import defaultdict
from threading import RLock
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.chat_graph import build_chat_graph
from chat_service import build_handlers

import os
import secrets

app = FastAPI(
    title="Agri-Food Supply Chain AI API",
    description="API untuk mengakses Gemini + LangGraph agent yang digunakan proyek.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

def verify_api_key(x_api_key: str = Header(default="")):
    expected_key = os.getenv("API_ACCESS_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API_ACCESS_KEY belum dikonfigurasi."
        )

    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="API key tidak valid."
        )

    return True

# One compiled graph per process; session history is kept separately.
graph = build_chat_graph()
handlers = build_handlers()
_sessions = defaultdict(list)
_sessions_lock = RLock()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    status: str
    session_id: str
    intent: str
    answer: str


def _invoke_agent(message: str, session_id: str):
    with _sessions_lock:
        history = _sessions[session_id]
        previous_question = history[-1]["question"] if history else ""
        previous_intent = history[-1]["intent"] if history else "general"

    # Keep compatibility with chat_graph.py's previous-question state.
    result = graph.invoke({
        "question": message,
        "handlers": handlers,
        "previous_question": previous_question,
        "previous_intent": previous_intent,
    })

    answer = str(result.get("answer") or "Belum ada jawaban yang dihasilkan.")
    intent = str(result.get("intent") or "general")

    with _sessions_lock:
        _sessions[session_id].append({
            "question": message,
            "intent": intent,
            "answer": answer,
        })
        # Limit memory per session.
        _sessions[session_id] = _sessions[session_id][-20:]

    return intent, answer


@app.get("/")
def root():
    return {
        "service": "Agri-Food Supply Chain AI API",
        "status": "running",
        "docs": "/docs",
        "chat_endpoint": "POST /api/ai/chat",
    }


@app.get("/api/ai/health")
def health():
    return {
        "status": "ok",
        "graph_loaded": graph is not None,
        "handlers": sorted(handlers.keys()),
        "model_provider": "Google Gemini (configured in chat_service.py)",
    }


@app.post(
        "/api/ai/chat", 
        response_model=ChatResponse,
        dependencies=[Depends(verify_api_key)])

def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message tidak boleh kosong.")

    session_id = request.session_id or str(uuid4())

    try:
        intent, answer = _invoke_agent(message, session_id)
        return ChatResponse(
            status="success",
            session_id=session_id,
            intent=intent,
            answer=answer,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memproses permintaan AI: {type(exc).__name__}: {exc}",
        )


@app.get("/api/ai/session/{session_id}")
def get_session(session_id: str):
    with _sessions_lock:
        messages = list(_sessions.get(session_id, []))
    return {"session_id": session_id, "message_count": len(messages), "messages": messages}


@app.delete("/api/ai/session/{session_id}")
def clear_session(session_id: str):
    with _sessions_lock:
        existed = session_id in _sessions
        _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id, "session_existed": existed}
