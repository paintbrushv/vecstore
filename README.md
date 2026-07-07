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

### Optional extras

The store itself needs only `sqlite-vec`. Everything else is opt-in via extras
declared in `pyproject.toml`:

```bash
pip install "vecstore[fastembed]"   # default local ONNX embedder (bge-small-en-v1.5, 384-dim)
pip install "vecstore[mcp]"         # MCP stdio server for `ask.py` (fastmcp)
pip install "vecstore[evals]"       # golden-set eval harness (pyyaml)
pip install -e ".[dev]"             # pytest + fastembed + pyyaml, for development
```

The package ships two top-level modules — `vecstore` and `ask` (`py-modules` in
`pyproject.toml`) — plus a `vecstore` console script (`vecstore … == python vecstore.py …`).

On a box without `ensurepip` (so `python -m venv` fails), create the venv with
[uv](https://github.com/astral-sh/uv):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[mcp,evals]'
```

Run the tests — they use `HashingEmbedder`, so no model download and no network:

```bash
python -m pytest -q
```

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

## Section-aware chunking

Split markdown-ish text into per-section chunks before indexing, so retrieval
and citations land on a section rather than a whole file:

```python
from vecstore import chunk_markdown, Chunk

chunks = chunk_markdown(text, path, max_tokens=400, overlap=50)   # -> list[Chunk]
```

`Chunk` is a dataclass:

```python
@dataclass
class Chunk:
    key: str    # "{path}#{section-slug}-{i}"  (i = window index within that section slug)
    text: str
    meta: dict  # {"path", "section", "date", "source"}
```

Section boundaries are detected three ways:

- **Markdown headings** — `#` through `######`.
- **Banner blocks** — a rule line of 4+ `=`, `─`, or `-` adjacent to a title
  line promotes that title to a heading (the line above wins, setext-style;
  otherwise the line below).
- **`── <title>` rule lines** — the title is the line stripped of its leading
  rule characters and whitespace.

Behavior worth knowing:

- A heading with no body emits **no chunk**; text before the first boundary
  becomes section `"preamble"`.
- Oversize sections split on paragraph boundaries into windows of
  `<= max_tokens` (token estimate = characters / 4), carrying `>= overlap`
  tokens from the previous window.
- `meta["date"]` = first `YYYY-MM-DD` in the filename, else the file's mtime
  date (UTC), else `""`. `meta["source"]` = the parent directory name (or
  `"root"`). `meta["section"]` = the section title.

## Chunked ingest

`sync_chunks` upserts one file's chunks and prunes stale ones in a single call:

```python
store.sync_chunks(path, chunks)
# -> {"added": int, "updated": int, "skipped": int, "deleted": int}
```

It deletes any stored key under `"{path}#…"` that is absent from the new chunk
set (a section that was moved or removed), and also deletes a stale whole-file
doc keyed exactly `"{path}"` — so migrating a file from whole-file to chunked
ingest cleans up the old row automatically.

From the CLI, `--chunk` ingests by section instead of whole files, and
`--source NAME` overrides the derived `meta["source"]`:

```bash
# section-chunk every markdown file under content/ and tag the source
python vecstore.py --db index.db ingest 'content/**/*.md' --chunk --source spot-desk
```

Without `--chunk`, `ingest` upserts each file as one document (keyed by path);
`--source` then sets `meta = {"path": p, "source": NAME}`.

### Cheap metadata refresh

When a document's **text** is unchanged but its **meta** changed, the store
updates the stored meta in place and still returns `"skipped"` — no re-embed,
the vector table is untouched. Re-tagging a corpus (e.g. a new `source`) is
therefore free.

## Filtered search

`search` accepts an optional post-retrieval metadata filter:

```python
store.search(query, k=5, where=None)
```

Supported `where` keys (any other key raises `ValueError`):

| key | matches |
| --- | --- |
| `date_from` | `meta["date"] >= value` (inclusive ISO string compare) |
| `date_to` | `meta["date"] <= value` (inclusive ISO string compare) |
| `source` | `meta["source"] == value` |
| `path_prefix` | `meta["path"].startswith(value)` |

When `where` is set the store overfetches `k * 4` candidates, then filters and
truncates to `k`. A document **missing** the meta key a filter targets never
matches that filter.

```python
store.search("q3 revenue", k=5, where={"source": "reports", "date_from": "2025-01-01"})
```

## Cited answers (RAG)

`ask.py` layers numbered-citation prompting on top of retrieval:

```python
from ask import Ask

rag = Ask(store, llm=None)                       # llm defaults to $VECSTORE_ASK_CMD
out = rag.ask("what changed in q3?", k=8, where=None)
# out == {"answer": str, "citations": [{"n", "key", "path", "section", "date"}]}
```

The default LLM shells out to `$VECSTORE_ASK_CMD`, passing the built prompt on
stdin and reading the answer from stdout:

- `VECSTORE_ASK_CMD` — the command to run (e.g. `claude -p`). **Unset → the call
  raises `RuntimeError`.** Pass your own `llm=` callable to skip the shell.
- `VECSTORE_ASK_TIMEOUT` — seconds before the subprocess is killed (default `300`).

### `ask` CLI

```bash
python vecstore.py --db index.db ask "what changed in q3?" -k 8 \
    [--source S] [--date-from D] [--path-prefix P] [--json]
```

`ask.py` also has its own CLI (which additionally supports `--date-to`):

```bash
python ask.py --db index.db ask "what changed in q3?" -k 8 \
    [--source S] [--date-from D] [--date-to D] [--path-prefix P] [--json]
```

Without `--json` both print the answer followed by a numbered citation list.

### MCP server

Serve `rag_ask(question, k=8)` over MCP stdio (needs the `mcp` extra; FastMCP is
imported lazily, so nothing is required until you actually serve):

```bash
python ask.py --db index.db serve-mcp
```

## Eval harness (golden set)

`evals/run_golden.py` grades retrieval (and optionally answers) against a YAML
golden set, and doubles as an adversarial / policy gate:

```bash
python evals/run_golden.py --db D --gold G.yaml --k 8 \
    [--ask] [--threshold N] [--embedder hashing|fastembed] \
    [--adversarial] [--deny-file F] [--allow-prefix P ...]
```

Flags:

- `--k` — retrieval depth for hit@k (default `8`; `--adversarial` forces `k=20`).
- `--ask` — also grade LLM answers (uses `$VECSTORE_ASK_CMD`): every `[n]` the
  answer cites must be a valid citation, and each `must_contain` string must
  appear in the answer.
- `--threshold N` — minimum passing questions for exit `0` (default: all must pass).
- `--embedder hashing|fastembed` — open the store with the matching embedder.
- `--adversarial` — an entry PASSES iff nothing forbidden is retrieved at `k=20`.
- `--deny-file F` — file of forbidden literal terms (one per line,
  case-insensitive), scanned across the whole docs table **and** every retrieval.
- `--allow-prefix P` — repeatable key path-prefix allowlist; any stored key
  whose path part starts with none of the prefixes is a violation.

**Exit codes:**

- `0` — passing questions `>= threshold`.
- `1` — golden score below threshold.
- `2` — a deny term matched, or a key fell outside the `--allow-prefix` set.
  This takes precedence over `0`/`1`.

**Gold entry schema** (YAML list; every field except `question` is optional):

```yaml
- question: "how much VRAM does the 3090 have?"
  must_cite_paths: ["content/gpu/"]   # hit@k = a top-k hit whose meta['path'] startswith a prefix
  must_contain: ["24GB"]              # (with --ask) substrings the answer must include
  forbid_path_prefixes: ["drafts/"]   # (with --adversarial) paths that must NOT be retrieved
  forbid_section_slugs: ["internal"]  # section slugs that must NOT be retrieved
  forbid_text_regex: "secret|token"   # regex that must NOT match any retrieved text
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

### Nightly ingest

Each corpus repo runs its own cron, then re-indexes with one command:

```bash
python vecstore.py --db index.db ingest '<glob>' --chunk
```

Because upserts are content-hashed, only new or changed docs are re-embedded, so
a nightly run over an otherwise-unchanged corpus is nearly free.

## What this is not

Not a server, not a graph code-search tool (that's CodeGraph), not a distributed
vector DB. It's the smallest thing that gives a single app durable, local,
meaning-based search over its own text.
