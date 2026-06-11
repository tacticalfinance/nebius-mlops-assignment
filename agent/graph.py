"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import APIConnectionError

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema_for_question

# Total generate + revise calls before the loop is forced to stop.
# Phase 6 tuning: cut 3 -> 2. The H100 baseline eval measured per-iteration
# pass rate at iter_0=43.3%, iter_1=50%, iter_2=50% - the second revise adds
# zero accuracy while chaining two more sequential vLLM calls (revise+verify)
# onto the worst-case agent latency path.
MAX_ITERATIONS = 2

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
VERIFY_EXECUTION_MAX_CHARS = 700
REVISE_EXECUTION_MAX_CHARS = 500
REVISE_PRIOR_ATTEMPTS_MAX_CHARS = 700
REVISE_SQL_MAX_CHARS = 900
VERIFY_SQL_MAX_CHARS = 1200
GENERATE_MAX_TOKENS = 256
REVISE_MAX_TOKENS = 256
VERIFY_MAX_TOKENS = 96


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


@lru_cache(maxsize=1)
def llm() -> ChatOpenAI:
    """Shared chat client pointed at VLLM_BASE_URL (your local vLLM by default).

    Cached as a singleton so its underlying httpx connection pool is reused
    across every node call. A fresh ChatOpenAI per call meant a fresh TCP
    connection per LLM call - under load that connection churn was itself
    producing the APIConnectionErrors _invoke retries on.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        # Safe default for this workload. Per-node overrides below can go
        # lower, but this guards against oversized completion budgets if a
        # call path forgets to pass one explicitly.
        max_tokens=256,
        timeout=600,
        max_retries=2,
    )


def _invoke(
    messages: list[tuple[str, str]],
    *,
    max_tokens: int,
    attempts: int = 4,
) -> BaseMessage:
    """llm().invoke() with retry on transient connection drops.

    Under load, connection drops here are sub-second blips (vLLM's own
    per-call lifecycle is ~2-6s) - a 20/40/60s backoff was 10-100x longer
    than the blip itself, and that dead time tied up the agent's threadpool
    long enough to cause cascading connection failures upstream.
    """
    last_exc: APIConnectionError | None = None
    for attempt in range(attempts):
        try:
            return llm().bind(max_tokens=max_tokens).invoke(messages)
        except APIConnectionError as e:
            last_exc = e
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
    assert last_exc is not None
    raise last_exc


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema_for_question(state.db_id, state.question)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of an LLM reply, defensively.

    Models often wrap the requested JSON in prose or markdown fences, or emit
    extra text around it. Try a fenced block first, then the first balanced
    `{...}` span, then the raw text - parsing whichever succeeds first.
    """
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    candidates.append(text)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Parse model-emitted booleans without treating "false" as truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _recent_sql_attempts(history: list[dict[str, Any]]) -> str:
    """Render prior generate/revise attempts so the reviser can avoid repeats."""
    attempts = [h.get("sql", "").strip() for h in history if h.get("node") in ("generate_sql", "revise")]
    attempts = [sql for sql in attempts if sql]
    if not attempts:
        return "None."

    lines: list[str] = []
    for idx, sql in enumerate(attempts, 1):
        compact = " ".join(sql.split())
        lines.append(f"{idx}. {compact}")
    return "\n".join(lines)


def _clip_text(text: str, max_chars: int) -> str:
    """Keep prompt fields bounded so revise loops cannot exceed context limits."""
    compact = text.strip()
    if len(compact) <= max_chars:
        return compact
    keep = max_chars - 32
    if keep <= 0:
        return compact[:max_chars]
    return compact[:keep] + "\n...[truncated for prompt budget]"


def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = _invoke(
        [
            ("system", prompts.GENERATE_SQL_SYSTEM),
            ("user", prompts.GENERATE_SQL_USER.format(
                schema=state.schema,
                question=state.question,
            )),
        ],
        max_tokens=GENERATE_MAX_TOKENS,
    )
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """
    execution = state.execution
    rendered = execution.render(max_rows=5) if execution is not None else "ERROR: agent produced no execution result"
    rendered = _clip_text(rendered, VERIFY_EXECUTION_MAX_CHARS)
    sql_for_verify = _clip_text(state.sql, VERIFY_SQL_MAX_CHARS)

    response = _invoke(
        [
            ("system", prompts.VERIFY_SYSTEM),
            ("user", prompts.VERIFY_USER.format(
                question=state.question,
                sql=sql_for_verify,
                execution=rendered,
            )),
        ],
        max_tokens=VERIFY_MAX_TOKENS,
    )
    parsed = _extract_json_object(response.content)

    # Be defensive: an execution error is never plausible regardless of what
    # the verifier says, and a missing/malformed verdict should not be treated
    # as success (fail closed -> let the loop try to revise).
    if execution is None or not execution.ok:
        ok = False
        issue = parsed.get("issue") or (execution.error if execution else "agent produced no execution result")
    elif not parsed:
        # A malformed verifier reply should not force a rewrite of an otherwise
        # plausible, non-empty answer. Fail closed only for empty results.
        ok = execution.row_count > 0
        issue = "" if ok else "verifier response could not be parsed and query returned 0 rows"
    else:
        ok = _coerce_bool(parsed.get("ok"), default=False)
        issue = str(parsed.get("issue", "") or "")
        if not ok and execution.row_count > 0 and not issue:
            ok = True

    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    execution = state.execution
    rendered = execution.render(max_rows=3) if execution is not None else "ERROR: agent produced no execution result"
    rendered = _clip_text(rendered, REVISE_EXECUTION_MAX_CHARS)
    prior_attempts = _clip_text(_recent_sql_attempts(state.history), REVISE_PRIOR_ATTEMPTS_MAX_CHARS)
    sql_for_revise = _clip_text(state.sql, REVISE_SQL_MAX_CHARS)

    response = _invoke(
        [
            ("system", prompts.REVISE_SYSTEM),
            ("user", prompts.REVISE_USER.format(
                schema=state.schema,
                question=state.question,
                sql=sql_for_revise,
                execution=rendered,
                issue=state.verify_issue,
                prior_attempts=prior_attempts,
            )),
        ],
        max_tokens=REVISE_MAX_TOKENS,
    )
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "revise", "sql": sql}],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
