"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _tokenize(text: str) -> set[str]:
    parts = re.findall(r"[A-Za-z0-9_]+", text)
    tokens: set[str] = set()
    for part in parts:
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part.replace("_", " "))
        for token in expanded.lower().split():
            if len(token) >= 2:
                tokens.add(token)
                if token.endswith("s") and len(token) >= 4:
                    tokens.add(token[:-1])
    return tokens


@lru_cache(maxsize=32)
def _schema_catalog(db_id: str) -> list[dict]:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    catalog: list[dict] = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            col_lines: list[str] = []
            column_names: list[str] = []
            neighbors: set[str] = set()
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                column_names.append(name)
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                neighbors.add(fk[2])
                if fk[4] is None:
                    col_lines.append(
                        f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}"
                    )
                else:
                    col_lines.append(
                        f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4])})"
                    )
            ddl = "\n".join([
                f"CREATE TABLE {_q(t)} (",
                ",\n".join(col_lines),
                ");",
            ])
            search_text = " ".join([t, *column_names]).replace("_", " ")
            catalog.append({
                "table": t,
                "ddl": ddl,
                "neighbors": neighbors,
                "tokens": _tokenize(search_text),
            })
    return catalog


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    parts: list[str] = [f"-- Database: {db_id}"]
    for entry in _schema_catalog(db_id):
        parts.append("")
        parts.append(entry["ddl"])
    return "\n".join(parts)


def render_schema_for_question(
    db_id: str,
    question: str,
    *,
    max_chars: int = 2600,
    min_tables: int = 2,
    max_tables: int = 6,
) -> str:
    """Render only the most relevant tables for a question.

    This keeps vLLM prompts under the model context limit during revise loops,
    where the schema, previous SQL attempts, and execution output all compete
    for prompt budget.
    """
    full = render_schema(db_id)
    if len(full) <= max_chars:
        return full

    catalog = _schema_catalog(db_id)
    question_tokens = _tokenize(question.replace("-", " ").replace("/", " "))
    scored: list[tuple[int, dict]] = []
    for entry in catalog:
        overlap = len(question_tokens & entry["tokens"])
        table_name = entry["table"].lower()
        table_tokens = _tokenize(table_name)
        bonus = 2 if table_name in question.lower() else 0
        bonus += len(question_tokens & table_tokens)
        scored.append((overlap + bonus, entry))
    scored.sort(key=lambda item: (item[0], -len(item[1]["ddl"])), reverse=True)

    selected: list[dict] = []
    selected_names: set[str] = set()
    for score, entry in scored:
        if score <= 0 and len(selected) >= min_tables:
            break
        if entry["table"] in selected_names:
            continue
        selected.append(entry)
        selected_names.add(entry["table"])
        if len(selected) >= max_tables:
            break

    if not selected:
        selected = [entry for _score, entry in scored[:min(max_tables, len(scored))]]
        selected_names = {entry["table"] for entry in selected}

    # Pull in direct FK neighbors so join paths still exist.
    for entry in list(selected):
        for neighbor in entry["neighbors"]:
            if neighbor in selected_names:
                continue
            match = next((item for item in catalog if item["table"] == neighbor), None)
            if match is not None and len(selected) < max_tables:
                selected.append(match)
                selected_names.add(match["table"])

    parts = [f"-- Database: {db_id}", "-- Schema pruned to likely relevant tables for this question."]
    for entry in selected:
        candidate = "\n".join(parts + ["", entry["ddl"]])
        if len(candidate) > max_chars and len(parts) > 2:
            break
        parts.append("")
        parts.append(entry["ddl"])

    # Fall back to the first few tables if the selected subset still ended up empty.
    if len(parts) <= 2:
        for entry in catalog[:min_tables]:
            parts.append("")
            parts.append(entry["ddl"])
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
