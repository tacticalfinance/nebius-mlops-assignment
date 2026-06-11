"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    Per-iteration SQL is pulled from the agent's history:
      history entries with node == 'generate_sql' or 'revise' each carry a
      'sql' key; we score them in order (iter 0, iter 1, ...).
    Carry-forward within this result: if the agent stopped at iter j, slots
    j+1 .. max are filled with the same correctness as iter j so that
    summarize() can aggregate uniformly across questions with different
    iteration counts.
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    # Gold rows — run once; treat as None if gold SQL errors
    _gold_ok, gold_rows, _gold_err = run_sql(db_id, gold_sql)
    if not _gold_ok:
        gold_rows = None

    # Call the agent
    agent_sql = ""
    agent_iters = 0
    agent_ok = False
    agent_error: str | None = None
    history: list[dict] = []

    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id,
                  "tags": {"db_id": db_id, "phase": "eval"}},
            timeout=1800.0,
        )
        if resp.status_code == 200:
            body = resp.json()
            agent_sql = body.get("sql", "")
            agent_iters = body.get("iterations", 0)
            agent_ok = body.get("ok", False)
            agent_error = body.get("error")
            history = body.get("history", [])
        else:
            agent_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"

    # Extract per-iteration SQLs from history in order
    sql_steps = [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise") and "sql" in h]

    # If the response succeeded but history is empty (e.g. server swallowed it),
    # fall back to just scoring the final sql as iteration 0.
    if not sql_steps and agent_sql:
        sql_steps = [agent_sql]

    # Score each iteration
    per_iter: list[bool] = []
    for sql in sql_steps:
        _ok, pred_rows, _err = run_sql(db_id, sql)
        per_iter.append(matches(gold_rows, pred_rows) if _ok else False)

    # Carry-forward to MAX_ITERATIONS slots so summarize() gets a uniform-length list
    from agent.graph import MAX_ITERATIONS  # avoid circular at module level
    max_slots = MAX_ITERATIONS
    if per_iter:
        last = per_iter[-1]
        while len(per_iter) < max_slots:
            per_iter.append(last)
    else:
        # Agent produced nothing useful; all iterations incorrect
        per_iter = [False] * max_slots

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": gold_sql,
        "agent_sql": agent_sql,
        "iterations": agent_iters,
        "agent_ok": agent_ok,
        "agent_error": agent_error,
        "per_iter_correct": per_iter,
        "final_correct": per_iter[-1] if per_iter else False,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"total": 0, "overall_pass_rate": 0.0, "per_iter_pass_rate": {}}

    # Number of slots = length of per_iter_correct (all results should match
    # since eval_one pads to MAX_ITERATIONS)
    n_slots = max(len(r["per_iter_correct"]) for r in results)

    # Per-iteration pass rate: for each slot k, fraction of questions correct
    # using iteration-k SQL (carry-forward already applied per eval_one)
    per_iter_pass: dict[str, float] = {}
    for k in range(n_slots):
        correct = sum(
            r["per_iter_correct"][k] if k < len(r["per_iter_correct"]) else r["per_iter_correct"][-1]
            for r in results
        )
        per_iter_pass[f"iter_{k}"] = correct / n

    overall = sum(r["final_correct"] for r in results) / n

    return {
        "total": n,
        "overall_pass_rate": overall,
        "per_iter_pass_rate": per_iter_pass,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
