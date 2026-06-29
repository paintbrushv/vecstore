"""vecstore — a tiny, reusable local semantic-search store on SQLite + sqlite-vec.

Drop this single file into any project. It gives you local, offline semantic
search backed by one SQLite file:

    from vecstore import VecStore
    store = VecStore("index.db")            # uses the default local embedder
    store.upsert("doc-1", "RTX 3090 has 24GB of VRAM")
    store.upsert("doc-2", "Apple M3 Max uses unified memory")
    for hit in store.search("how much memory does the 3090 have?", k=3):
        print(hit["key"], round(hit["score"], 3), hit["text"][:60])

Design goals (see README):
- **Local-first.** Embeddings run on-device (fastembed/ONNX by default — no torch,
  no remote API). The same model runs on your Mac mini and on an OVH VPS.
- **Incremental + deterministic.** Each doc is keyed and content-hashed, so
  re-running an ingest only re-embeds changed docs. Same input + same model =>
  same vectors, which makes prod sync cheap (ship the .db OR rebuild on the VPS).
- **Pluggable embedder.** Swap fastembed for sentence-transformers, llama.cpp,
  or an API by passing any object with `.embed(list[str]) -> list[list[float]]`
  and a `.model_id` string. Vectors are unit-normalized here, so L2 ranking
  equals cosine ranking regardless of the backend or sqlite-vec version.

Requires: `pip install sqlite-vec` (tiny). The default embedder also needs
`pip install fastembed`; without it, pass your own embedder or use the
dependency-free HashingEmbedder (good for tests/offline, weak semantics).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Protocol, Sequence


# ---------------------------------------------------------------------------
# Embedder interface
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


def _unit_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


class FastEmbedEmbedder:
    """Default local embedder. ONNX-based, CPU-friendly, no torch, no network
    calls at inference (the model downloads once on first use, then is cached).

    bge-small-en-v1.5 is 384-dim and a strong speed/quality tradeoff for
    "lots of small projects." Pass a different model_name for larger/multilingual.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - import-guard
            raise ImportError(
                "FastEmbedEmbedder needs fastembed: `pip install fastembed`. "
                "Or pass your own embedder / use HashingEmbedder for offline tests."
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.model_name = model_name
        self.model_id = f"fastembed:{model_name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # fastembed returns numpy arrays; normalize for cosine-via-L2 ranking.
        return [_unit_normalize(v.tolist()) for v in self._model.embed(list(texts))]


class HashingEmbedder:
    """Dependency-free, deterministic embedder using the hashing trick.

    Semantics are weak (no learned meaning) but it is fast, offline, needs zero
    installs, and is fully deterministic — ideal for unit tests, CI, and a
    graceful fallback when no model is available. Do NOT use for real semantic
    search quality; swap in FastEmbedEmbedder for that.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_id = f"hashing:{dim}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                bucket = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[bucket] += sign
            out.append(_unit_normalize(vec))
        return out


# ---------------------------------------------------------------------------
# Core store
# ---------------------------------------------------------------------------


def _serialize_f32(vec: Sequence[float]) -> bytes:
    """Pack a float vector into the compact little-endian blob vec0 expects."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VecStore:
    """A keyed, content-hashed semantic store over one SQLite file.

    Args:
        path:     SQLite file path (use ":memory:" for ephemeral/test stores).
        embedder: Any Embedder. Defaults to FastEmbedEmbedder() lazily on first
                  embed so importing this module never forces fastembed.
        table:    Logical name; the doc + vec tables are derived from it.
    """

    def __init__(
        self,
        path: str,
        embedder: Optional[Embedder] = None,
        table: str = "docs",
    ) -> None:
        import sqlite3

        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover - import-guard
            raise ImportError(
                "vecstore needs sqlite-vec: `pip install sqlite-vec`."
            ) from exc

        self.path = path
        self._embedder = embedder
        self._docs = table
        self._vec = f"vec_{table}"
        self._dim: Optional[int] = None

        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_meta()

    # ---- embedder (lazy default) ------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = FastEmbedEmbedder()
        return self._embedder

    # ---- schema -----------------------------------------------------------

    def _init_meta(self) -> None:
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS vecstore_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.db.commit()

    def _ensure_schema(self, dim: int) -> None:
        """Create the doc + vec tables once we know the embedding dimension.

        Guards against silently mixing models/dimensions in one DB — if a stored
        model_id/dim disagrees with the active embedder, raise rather than
        corrupt the index. Use `rebuild()` after an intentional model change.
        """
        stored_dim = self._meta_get("dim")
        stored_model = self._meta_get("model_id")
        active_model = self.embedder.model_id

        if stored_dim is not None:
            if int(stored_dim) != dim or stored_model != active_model:
                raise RuntimeError(
                    f"Embedder mismatch for {self.path}: stored "
                    f"({stored_model}, dim={stored_dim}) != active "
                    f"({active_model}, dim={dim}). Run rebuild() to re-embed, "
                    f"or open with the original embedder."
                )
            self._dim = dim
            return

        self.db.execute(
            f"CREATE TABLE IF NOT EXISTS {self._docs} ("
            " id INTEGER PRIMARY KEY,"
            " key TEXT UNIQUE NOT NULL,"
            " text TEXT NOT NULL,"
            " meta TEXT,"
            " content_hash TEXT NOT NULL,"
            " updated_at TEXT NOT NULL)"
        )
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec} "
            f"USING vec0(embedding float[{dim}])"
        )
        self._meta_set("dim", str(dim))
        self._meta_set("model_id", active_model)
        self.db.commit()
        self._dim = dim

    def _meta_get(self, key: str) -> Optional[str]:
        row = self.db.execute(
            "SELECT value FROM vecstore_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _meta_set(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO vecstore_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ---- writes -----------------------------------------------------------

    def upsert(self, key: str, text: str, meta: Optional[dict] = None) -> str:
        """Insert or update one document. Returns "added", "updated", or
        "skipped" (skipped == content unchanged, no re-embed)."""
        return self.upsert_many([(key, text, meta)])[0]

    def upsert_many(
        self, items: Iterable[tuple[str, str, Optional[dict]]]
    ) -> list[str]:
        items = list(items)
        if not items:
            return []

        # Figure out which docs actually changed before spending any embed cost.
        results: list[Optional[str]] = [None] * len(items)
        to_embed: list[int] = []
        for i, (key, text, _meta) in enumerate(items):
            h = _content_hash(text)
            row = self.db.execute(
                f"SELECT id, content_hash FROM {self._docs} WHERE key = ?", (key,)
            ).fetchone() if self._dim is not None or self._schema_exists() else None
            if row is not None and row["content_hash"] == h:
                results[i] = "skipped"
            else:
                results[i] = "updated" if row is not None else "added"
                to_embed.append(i)

        if not to_embed:
            return [r or "skipped" for r in results]

        vectors = self.embedder.embed([items[i][1] for i in to_embed])
        if self._dim is None:
            self._ensure_schema(len(vectors[0]))
        if len(vectors[0]) != self._dim:
            raise RuntimeError(
                f"Embedder returned dim {len(vectors[0])}, store expects {self._dim}."
            )

        for vec, i in zip(vectors, to_embed):
            key, text, meta = items[i]
            h = _content_hash(text)
            meta_json = json.dumps(meta) if meta is not None else None
            existing = self.db.execute(
                f"SELECT id FROM {self._docs} WHERE key = ?", (key,)
            ).fetchone()
            if existing is not None:
                doc_id = existing["id"]
                self.db.execute(
                    f"UPDATE {self._docs} SET text=?, meta=?, content_hash=?, "
                    f"updated_at=? WHERE id=?",
                    (text, meta_json, h, _now_iso(), doc_id),
                )
                self.db.execute(f"DELETE FROM {self._vec} WHERE rowid=?", (doc_id,))
            else:
                cur = self.db.execute(
                    f"INSERT INTO {self._docs}(key, text, meta, content_hash, updated_at)"
                    f" VALUES(?,?,?,?,?)",
                    (key, text, meta_json, h, _now_iso()),
                )
                doc_id = cur.lastrowid
            self.db.execute(
                f"INSERT INTO {self._vec}(rowid, embedding) VALUES(?, ?)",
                (doc_id, _serialize_f32(vec)),
            )
        self.db.commit()
        return [r for r in results]  # type: ignore[return-value]

    def delete(self, key: str) -> bool:
        row = self.db.execute(
            f"SELECT id FROM {self._docs} WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return False
        self.db.execute(f"DELETE FROM {self._vec} WHERE rowid = ?", (row["id"],))
        self.db.execute(f"DELETE FROM {self._docs} WHERE id = ?", (row["id"],))
        self.db.commit()
        return True

    def rebuild(self) -> int:
        """Re-embed every stored doc with the current embedder. Use after an
        intentional model change. Returns the number of docs re-embedded."""
        if not self._schema_exists():
            return 0
        rows = self.db.execute(
            f"SELECT key, text, meta FROM {self._docs}"
        ).fetchall()
        # Drop vec table + meta so _ensure_schema rebuilds for the new model.
        self.db.execute(f"DROP TABLE IF EXISTS {self._vec}")
        self.db.execute(f"DELETE FROM {self._docs}")
        self.db.execute("DELETE FROM vecstore_meta WHERE key IN ('dim','model_id')")
        self.db.commit()
        self._dim = None
        items = [
            (r["key"], r["text"], json.loads(r["meta"]) if r["meta"] else None)
            for r in rows
        ]
        self.upsert_many(items)
        return len(items)

    # ---- reads ------------------------------------------------------------

    def search(
        self, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the k nearest docs as dicts with key/text/meta/distance/score.
        score is cosine similarity in [-1, 1] (1 == identical)."""
        if not self._schema_exists() or self._dim is None and not self._load_dim():
            return []
        qvec = self.embedder.embed([query])[0]
        # vec0 requires the LIMIT/k constraint on the KNN query itself, so do the
        # nearest-neighbor scan in a CTE, then join doc rows onto the matches.
        rows = self.db.execute(
            f"WITH matches AS ("
            f"  SELECT rowid, distance FROM {self._vec} "
            f"  WHERE embedding MATCH ? ORDER BY distance LIMIT ?"
            f") "
            f"SELECT d.key, d.text, d.meta, m.distance "
            f"FROM matches m JOIN {self._docs} d ON d.id = m.rowid "
            f"ORDER BY m.distance",
            (_serialize_f32(qvec), k),
        ).fetchall()
        out = []
        for r in rows:
            dist = float(r["distance"])
            # Unit vectors => L2^2 = 2(1 - cos). Recover cosine similarity.
            cosine = 1.0 - (dist * dist) / 2.0
            out.append(
                {
                    "key": r["key"],
                    "text": r["text"],
                    "meta": json.loads(r["meta"]) if r["meta"] else None,
                    "distance": dist,
                    "score": round(cosine, 6),
                }
            )
        return out

    def stats(self) -> dict[str, Any]:
        n = 0
        if self._schema_exists():
            n = self.db.execute(
                f"SELECT COUNT(*) AS n FROM {self._docs}"
            ).fetchone()["n"]
        return {
            "path": self.path,
            "docs": n,
            "model_id": self._meta_get("model_id"),
            "dim": self._meta_get("dim"),
        }

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "VecStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- internals --------------------------------------------------------

    def _schema_exists(self) -> bool:
        row = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self._docs,),
        ).fetchone()
        return row is not None

    def _load_dim(self) -> bool:
        d = self._meta_get("dim")
        if d is not None:
            self._dim = int(d)
            return True
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_embedder(args) -> Optional[Embedder]:
    if args.embedder == "hashing":
        return HashingEmbedder()
    if args.embedder == "fastembed":
        return FastEmbedEmbedder(args.model)
    return None  # default (lazy fastembed)


def _cmd_ingest(args) -> int:
    import glob as globlib

    store = VecStore(args.db, embedder=_build_embedder(args))
    paths = sorted(globlib.glob(args.glob, recursive=True))
    if not paths:
        print(f"no files matched {args.glob!r}")
        return 1
    items = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", errors="ignore") as fh:
            items.append((p, fh.read(), {"path": p}))
    results = store.upsert_many(items)
    counts: dict[str, int] = {}
    for r in results:
        counts[r] = counts.get(r, 0) + 1
    print(f"ingested {len(items)} files: {counts}")
    print(store.stats())
    return 0


def _cmd_add(args) -> int:
    store = VecStore(args.db, embedder=_build_embedder(args))
    print(store.upsert(args.key, args.text))
    return 0


def _cmd_search(args) -> int:
    store = VecStore(args.db, embedder=_build_embedder(args))
    hits = store.search(args.query, k=args.k)
    for h in hits:
        print(f"{h['score']:.3f}  {h['key']}")
        if args.show_text:
            print(f"        {h['text'][:120].strip()}")
    if not hits:
        print("(no results)")
    return 0


def _cmd_rebuild(args) -> int:
    store = VecStore(args.db, embedder=_build_embedder(args))
    n = store.rebuild()
    print(f"re-embedded {n} docs -> {store.stats()}")
    return 0


def _cmd_stats(args) -> int:
    store = VecStore(args.db, embedder=_build_embedder(args))
    print(store.stats())
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="vecstore", description="Local semantic search on SQLite + sqlite-vec."
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

    sp = sub.add_parser("ingest", help="embed files matching a glob")
    sp.add_argument("glob", help="e.g. 'content/**/*.md'")
    sp.set_defaults(func=_cmd_ingest)

    sp = sub.add_parser("add", help="add/update one document")
    sp.add_argument("--key", required=True)
    sp.add_argument("--text", required=True)
    sp.set_defaults(func=_cmd_add)

    sp = sub.add_parser("search", help="semantic search")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=5)
    sp.add_argument("--show-text", action="store_true")
    sp.set_defaults(func=_cmd_search)

    sp = sub.add_parser("rebuild", help="re-embed all docs (after model change)")
    sp.set_defaults(func=_cmd_rebuild)

    sp = sub.add_parser("stats", help="print store stats")
    sp.set_defaults(func=_cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
