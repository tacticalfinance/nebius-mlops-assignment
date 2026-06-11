"""Schema-rendering helper.

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.

Each column is annotated with a `/* ... */` comment carrying (a) a short
description from BIRD's database_description CSVs when available and
(b) a few example values sampled from the data. Baseline eval showed the
dominant failure class was the model guessing literals it cannot see
('m' vs 'M', 'carcinogenic' vs '+', 'Calcium' vs 'ca', invented date /
lap-time formats) and cryptic columns (financial.A15). Without examples
the revise loop also has no signal to fix a zero-rows result.
"""
from __future__ import annotations

import csv
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"

# Per-column annotation budgets (chars). Kept tight so an annotated schema
# still fits the prompt budget in render_schema_for_question.
_MAX_DESC_CHARS = 80
_MAX_EXAMPLE_CHARS = 34
_MAX_EXAMPLES = 3


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _clean_text(value: str) -> str:
    return " ".join(value.replace("*/", "").split())


@lru_cache(maxsize=32)
def _column_descriptions(db_id: str) -> dict[tuple[str, str], str]:
    """Load BIRD column descriptions, keyed by (table_lower, column_lower).

    BIRD ships data/bird/.../<db_id>/database_description/<table>.csv with
    columns like original_column_name / column_description / value_description.
    Both texts matter: value_description is where e.g. '+' = carcinogenic or
    'normal range: 900 < N < 2000' lives. Missing files are fine - we just
    render without descriptions.
    """
    desc_dir = next(
        (d for d in DB_DIR.rglob("database_description") if d.parent.name == db_id),
        None,
    )
    if desc_dir is None:
        return {}

    out: dict[tuple[str, str], str] = {}
    for csv_path in desc_dir.glob("*.csv"):
        table = csv_path.stem.lower()
        rows: list[dict] = []
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                with csv_path.open(newline="", encoding=enc) as f:
                    rows = list(csv.DictReader(f))
                break
            except (UnicodeDecodeError, csv.Error):
                continue
        for row in rows:
            col = (row.get("original_column_name") or "").strip()
            if not col:
                continue
            parts = []
            for key in ("column_description", "value_description"):
                text = _clean_text(row.get(key) or "")
                if text and text.lower() != col.lower():
                    parts.append(text)
            desc = "; ".join(parts)
            if not desc:
                continue
            if len(desc) > _MAX_DESC_CHARS:
                desc = desc[: _MAX_DESC_CHARS - 3] + "..."
            out[(table, col.lower())] = desc
    return out


def _sample_values(conn: sqlite3.Connection, table: str, column: str) -> list[str]:
    """A few distinct example values for a column - text-ish ones only.

    Examples are what tell the model the data says 'M' not 'm', '+' not
    'carcinogenic', '1:27.452' not milliseconds. Long free-text columns
    (post bodies, titles) are skipped: their values don't generalize.
    """
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
            f"WHERE {_q(column)} IS NOT NULL AND {_q(column)} != '' LIMIT 6"
        ).fetchall()
    except sqlite3.Error:
        return []
    values: list[str] = []
    for (value,) in rows:
        if not isinstance(value, str):
            return []
        if len(value) > _MAX_EXAMPLE_CHARS:
            continue
        values.append(value)
        if len(values) >= _MAX_EXAMPLES:
            break
    return values


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
    descriptions = _column_descriptions(db_id)
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
                annotations: list[str] = []
                desc = descriptions.get((t.lower(), name.lower()))
                if desc:
                    annotations.append(desc)
                examples = _sample_values(conn, t, name)
                if examples:
                    annotations.append(
                        "e.g. " + ", ".join(f"'{v}'" for v in examples)
                    )
                if annotations:
                    line += " /* " + "; ".join(annotations) + " */"
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
    max_chars: int = 4400,
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
