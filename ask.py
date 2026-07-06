"""ask — cited RAG answers over a vecstore index.

Companion to vecstore.py (same repo, same drop-in philosophy). Kept out of
vecstore.py because answering adds an LLM dependency the core store must not
carry — mirroring vecstore's lazy-import pattern (fastembed, sqlite_vec).

    from vecstore import VecStore
    from ask import Ask

    store = VecStore("index.db")
    out = Ask(store).ask("what cleared below fair value this week?")
    print(out["answer"])
    for c in out["citations"]:
        print(f"[{c['n']}] {c['key']}")

The default LLM shells the command in $VECSTORE_ASK_CMD (e.g. `claude -p`)
with the fully built prompt on stdin and reads the answer from stdout. Pass
llm=callable for anything else (tests stub it — no network).

MCP: `python ask.py --db index.db serve-mcp` serves one tool
`rag_ask(question, k)` over stdio. FastMCP comes from the `[mcp]` extra and
is imported lazily inside the command, so the core install never needs it.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

from vecstore import FastEmbedEmbedder, HashingEmbedder, VecStore

# Grounding tail appended verbatim to every prompt. Contractual: the eval
# harness and downstream corpora match on this exact string — do not reword.
GROUNDING = (
    "Answer only from the provided context; cite sources inline as [n]. "
    "If the context does not contain the answer, say so."
)


def _default_llm(prompt: str) -> str:
    """Shell out to $VECSTORE_ASK_CMD with the prompt on stdin."""
    cmd = os.environ.get("VECSTORE_ASK_CMD")
    if not cmd:
        raise RuntimeError(
            "VECSTORE_ASK_CMD is not set. Export it (e.g. "
            "VECSTORE_ASK_CMD='claude -p') or pass llm= to Ask()."
        )
    import shlex
    import subprocess

    timeout = float(os.environ.get("VECSTORE_ASK_TIMEOUT", "300"))
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"VECSTORE_ASK_CMD ({cmd!r}) timed out after {timeout:g}s"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"VECSTORE_ASK_CMD not found: {cmd!r}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"VECSTORE_ASK_CMD exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


class Ask:
    """Retrieval + numbered-citation prompting over a VecStore."""

    def __init__(
        self, store: VecStore, llm: Callable[[str], str] | None = None
    ) -> None:
        self.store = store
        self.llm = llm if llm is not None else _default_llm

    def build_context(
        self, question: str, k: int, where: dict | None
    ) -> tuple[str, list[dict]]:
        """Numbered blocks '[n] {path} § {section} ({date})\\n{text}' in score
        order; returns (prompt, citations) with citations =
        [{n, key, path, section, date}]."""
        if where is not None:
            hits = self.store.search(question, k=k, where=where)
        else:
            hits = self.store.search(question, k=k)
        blocks: list[str] = []
        citations: list[dict] = []
        for n, hit in enumerate(hits, start=1):
            meta = hit.get("meta") or {}
            path = meta.get("path") or hit["key"].split("#", 1)[0]
            section = meta.get("section") or ""
            date = meta.get("date") or ""
            blocks.append(f"[{n}] {path} § {section} ({date})\n{hit['text']}")
            citations.append(
                {
                    "n": n,
                    "key": hit["key"],
                    "path": path,
                    "section": section,
                    "date": date,
                }
            )
        prompt = (
            "Context:\n\n"
            + "\n\n".join(blocks)
            + f"\n\nQuestion: {question}\n\n"
            + GROUNDING
        )
        return prompt, citations

    def ask(self, question: str, k: int = 8, where: dict | None = None) -> dict:
        """{"answer": str, "citations": [{n, key, path, section, date}]}"""
        prompt, citations = self.build_context(question, k, where)
        return {"answer": self.llm(prompt), "citations": citations}


# ---------------------------------------------------------------------------
# CLI (smoke use + MCP server; the full-featured CLI lives in `vecstore ask`)
# ---------------------------------------------------------------------------


def _stored_hashing_dim(db_path: str, default: int = 256) -> int:
    """Dim of the HashingEmbedder a store was built with (vecstore_meta
    model_id 'hashing:{dim}'), so `--embedder hashing` opens any
    hashing-built DB without a model-mismatch error."""
    import sqlite3

    try:
        db = sqlite3.connect(db_path)
        try:
            row = db.execute(
                "SELECT value FROM vecstore_meta WHERE key = 'model_id'"
            ).fetchone()
        finally:
            db.close()
    except sqlite3.Error:
        return default
    if row and row[0] and row[0].startswith("hashing:"):
        return int(row[0].split(":", 1)[1])
    return default


def _build_embedder(args):
    if args.embedder == "hashing":
        return HashingEmbedder(dim=_stored_hashing_dim(args.db))
    if args.embedder == "fastembed":
        return FastEmbedEmbedder(args.model)
    return None  # default (lazy fastembed)


def _cmd_ask(args) -> int:
    store = VecStore(args.db, embedder=_build_embedder(args))
    where: dict = {}
    if args.source:
        where["source"] = args.source
    if args.date_from:
        where["date_from"] = args.date_from
    if args.date_to:
        where["date_to"] = args.date_to
    if args.path_prefix:
        where["path_prefix"] = args.path_prefix
    out = Ask(store).ask(args.question, k=args.k, where=where or None)
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(out["answer"])
    if out["citations"]:
        print()
        for c in out["citations"]:
            print(f"[{c['n']}] {c['key']} ({c['date']})")
    return 0


def _cmd_serve_mcp(args) -> int:
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - import-guard
        raise ImportError(
            "serve-mcp needs fastmcp: `pip install 'vecstore[mcp]'`."
        ) from exc

    store = VecStore(args.db, embedder=_build_embedder(args))
    rag = Ask(store)
    mcp = FastMCP("vecstore")

    @mcp.tool()
    def rag_ask(question: str, k: int = 8) -> dict:
        """Answer a question from the local vecstore corpus with numbered
        citations: {"answer": str, "citations": [{n, key, path, section,
        date}]}."""
        return rag.ask(question, k=k)

    mcp.run()  # stdio transport (default)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="ask", description="Cited RAG answers over a vecstore index."
    )
    p.add_argument("--db", required=True, help="SQLite file path")
    p.add_argument(
        "--embedder",
        choices=["default", "fastembed", "hashing"],
        default="default",
        help="default/fastembed = local ONNX model; hashing = deps-free fallback",
    )
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="fastembed model")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ask", help="answer one question with citations")
    sp.add_argument("question")
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--source", help="filter: meta['source'] equality")
    sp.add_argument("--date-from", dest="date_from", help="filter: meta['date'] >= D (ISO)")
    sp.add_argument("--date-to", dest="date_to", help="filter: meta['date'] <= D (ISO)")
    sp.add_argument("--path-prefix", dest="path_prefix", help="filter: meta['path'] prefix")
    sp.add_argument("--json", action="store_true", help="print full result as JSON")
    sp.set_defaults(func=_cmd_ask)

    sp = sub.add_parser(
        "serve-mcp",
        help="serve rag_ask(question, k) over MCP stdio "
        "(needs `pip install 'vecstore[mcp]'`)",
    )
    sp.set_defaults(func=_cmd_serve_mcp)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
