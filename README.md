# vecstore

Tiny, reusable **local semantic search** for "lots of small projects." One
SQLite file, one Python file, a local embedding model (no torch, no API). Drop
it in, index some text, search it by meaning. Same code runs on your Mac mini
dev box and on an OVH VPS in prod.

- **Local-first.** Embeddings run on-device via [fastembed](https://github.com/qdrant/fastembed)
  (ONNX). No remote embedding API, no torch. Swap in any embedder you like.
- **One file, one DB.** Storage is [sqlite-vec](https://github.com/asg017/sqlite-vec).
  The whole index is a single `.db` — easy to ship, back up, or rsync.
- **Incremental + deterministic.** Docs are keyed and content-hashed, so re-ingest
  only re-embeds what changed. Same model + same text → same vectors, which is
  what makes prod sync cheap (see below).
- **MIT licensed** — clean for commercial use.

## Install

```bash
pip install sqlite-vec fastembed     # or: pip install -e ".[dev]"
```

`sqlite-vec` is required. `fastembed` is the default embedder (optional if you
pass your own). For tests/offline with zero ML deps, use `HashingEmbedder`.

## Use it as a library

```python
from vecstore import VecStore

store = VecStore("index.db")                      # default local embedder
store.upsert("gpu-3090", "The RTX 3090 has 24GB of GDDR6X VRAM.")
store.upsert("mac-m3",   "Apple M3 Max uses unified memory at 400 GB/s.")

for hit in store.search("how much memory does the 3090 have?", k=3):
    print(f"{hit['score']:.3f}  {hit['key']}  {hit['text'][:50]}")
```

`upsert` returns `"added" | "updated" | "skipped"`. `search` returns dicts with
`key`, `text`, `meta`, `distance`, and `score` (cosine similarity, 1.0 = identical).

### Bring your own embedder

Anything with `.embed(list[str]) -> list[list[float]]` and a `.model_id` string
works. Vectors are unit-normalized internally so L2 ranking equals cosine ranking
regardless of backend:

```python
store = VecStore("index.db", embedder=MyLlamaCppEmbedder())
```

Built-ins: `FastEmbedEmbedder(model_name=...)` (default) and `HashingEmbedder(dim=...)`
(deterministic, dependency-free, weak semantics — for tests/fallback).

## Use it from the CLI

```bash
# index every markdown file under content/ (keys = file paths)
python vecstore.py --db index.db ingest 'content/**/*.md'

# search
python vecstore.py --db index.db search "vram needed for a 13B model" -k 5 --show-text

# add one doc; print stats; re-embed everything after a model change
python vecstore.py --db index.db add --key note-1 --text "some text"
python vecstore.py --db index.db stats
python vecstore.py --db index.db rebuild

# offline / no model download (deterministic, weak quality)
python vecstore.py --db index.db --embedder hashing ingest 'docs/**/*.md'
```

## Syncing to prod (OVH VPS)

The index is just a `.db`, and embeddings are deterministic, so you have three
patterns — pick per project:

1. **Build-on-deploy (recommended default).** Ship only your content. On the VPS,
   run `python vecstore.py --db index.db ingest '<glob>'` in a post-deploy step.
   The content-hash means only new/changed docs get embedded — fast, and the DB
   is rebuilt server-side so it never lives in git. Needs the embedder available
   on the VPS (fastembed is pip-only, no GPU needed).
2. **Build-local, ship the `.db`.** Build the index on the Mac mini, then
   `rsync index.db vps:/app/`. Best for **mostly-static corpora** (article
   archives, docs) where prod shouldn't spend compute. No embedder needed in prod.
3. **Build-at-runtime.** App calls `upsert` as content is created. Simplest model;
   prod needs the embedder live.

After an intentional **model change**, run `rebuild()` (or the `rebuild` CLI) so
old and new vectors don't mix — the store refuses to mix models/dimensions in one
DB and will raise instead of silently corrupting the index.

## What this is not

Not a server, not a graph code-search tool (that's CodeGraph), not a distributed
vector DB. It's the smallest thing that gives a single app durable, local,
meaning-based search over its own text.
