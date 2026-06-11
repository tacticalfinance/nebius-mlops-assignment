"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402

_langfuse_enabled = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)
if _langfuse_enabled:
    from langfuse.langchain import CallbackHandler as _LFHandler


app = FastAPI()


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    callbacks: list[Any] = []
    if _langfuse_enabled:
        callbacks = [_LFHandler()]
    # `langfuse_tags` is a special metadata key the Langfuse LangChain handler
    # maps to real trace tags, so runs are filterable in the trace list
    # (e.g. phase:eval vs phase:load_test in Phase 6).
    metadata: dict[str, Any] = dict(req.tags)
    if req.tags:
        metadata["langfuse_tags"] = [f"{k}:{v}" for k, v in req.tags.items()]
    config: dict[str, Any] = {"callbacks": callbacks, "metadata": metadata}
    try:
        final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
