"""FastAPI backend integrating policy RAG and the refund agent."""

from __future__ import annotations

import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data_pipeline import DB_PATH, run_pipeline
from src.policy_rag import PolicyRAG
from src.refund_agent import RefundLedger, get_order, run_refund_agent

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Literal["mock"]
    policy_documents: int


class PolicyRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class RetrievedChunk(BaseModel):
    source: str
    chunk_index: int
    text: str
    score: float


class PolicyResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool
    retrieved: list[RetrievedChunk]


class RefundRequest(BaseModel):
    message: str = Field(min_length=3, max_length=1000)


class AgentStep(BaseModel):
    tool: str
    observation: dict[str, Any]


class RefundResponse(BaseModel):
    status: str
    final_answer: str
    steps: list[AgentStep]
    tokens_used: int


class DemoSessionRequest(BaseModel):
    profile: Literal["automatic_refund", "human_approval", "on_time_denial"]


class DemoSessionResponse(BaseModel):
    token: str
    profile: str
    order_id: str
    suggested_message: str


def ensure_database() -> None:
    """Build the ignored local database when starting from a clean clone."""
    if DB_PATH.is_file():
        try:
            with sqlite3.connect(DB_PATH) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            if {"customers", "orders", "products", "tickets"} <= tables:
                return
        except sqlite3.DatabaseError:
            pass
    run_pipeline()


ensure_database()
rag = PolicyRAG()
ledger = RefundLedger()
DEMO_PROFILES = {
    "automatic_refund": {"customer_id": "C0007", "order_id": "ORD-100048"},
    "human_approval": {"customer_id": "C0016", "order_id": "ORD-100145"},
    "on_time_denial": {"customer_id": "C0001", "order_id": "ORD-100001"},
}
demo_sessions: dict[str, dict[str, Any]] = {}
app = FastAPI(
    title="Instamart Customer Copilot API",
    version="1.0.0",
    description="Offline policy RAG and authenticated refund-agent API.",
)
origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        mode="mock",
        policy_documents=len({chunk["source"] for chunk in rag.chunks}),
    )


@app.post("/api/policy/query", response_model=PolicyResponse)
def query_policy(request: PolicyRequest) -> PolicyResponse:
    result = rag.answer(request.question)
    return PolicyResponse.model_validate(result)


@app.post("/api/demo/session", response_model=DemoSessionResponse)
def create_demo_session(request: DemoSessionRequest) -> DemoSessionResponse:
    """Issue an opaque token for one fixed synthetic demonstration identity."""
    profile = DEMO_PROFILES[request.profile]
    order = get_order(profile["order_id"], profile["customer_id"])
    if order.get("error"):
        raise HTTPException(status_code=503, detail="Demo order is unavailable.")
    order_time = datetime.fromisoformat(str(order["order_ts"])).replace(
        tzinfo=timezone.utc
    )
    delivery_minutes = int(order["delivery_minutes"] or 0)
    token = secrets.token_urlsafe(24)
    demo_sessions[token] = {
        "customer_id": profile["customer_id"],
        "request_time": order_time
        + timedelta(minutes=delivery_minutes, days=1),
    }
    return DemoSessionResponse(
        token=token,
        profile=request.profile,
        order_id=profile["order_id"],
        suggested_message=f"Refund order {profile['order_id']}",
    )


@app.post("/api/refunds/query", response_model=RefundResponse)
def query_refund(
    request: RefundRequest,
    authorization: Optional[str] = Header(default=None),
) -> RefundResponse:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="A demo session is required.")
    session = demo_sessions.get(authorization.removeprefix("Bearer ").strip())
    if session is None:
        raise HTTPException(status_code=401, detail="The demo session is invalid.")
    result = run_refund_agent(
        request.message,
        customer_id=session["customer_id"],
        ledger=ledger,
        request_time=session["request_time"],
    )
    return RefundResponse.model_validate(result)


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
