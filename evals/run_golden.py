"""run_golden — golden-set eval harness for a vecstore index.

Usage:
    python evals/run_golden.py --db index.db --gold golden.yaml --k 8 \\
        [--ask] [--threshold N] [--embedder hashing|fastembed] \\
        [--adversarial] [--deny-file terms.txt] [--allow-prefix P ...]

Gold entry schema (YAML list):
    - question: "strong buy breaker"
      must_cite_paths: ["logs/digests/"]        # retrieval hit@k prefixes
      must_contain: ["Square D"]                # --ask only: answer substrings
      forbid_path_prefixes: ["docs/private/"]   # --adversarial
      forbid_section_slugs: ["pricing"]         # --adversarial
      forbid_text_regex: "\\\\bSSN\\\\b"        # --adversarial

Scoring: hit@k = any top-k hit whose meta['path'] starts with any
must_cite_paths prefix. With --ask, additionally every [n] cited in the
answer must map to a returned citation and every must_contain string must
appear in the answer. --adversarial: an entry PASSES iff nothing forbidden
is retrieved at k=20.

Exit codes: 0 = pass; 1 = golden passes below threshold; 2 = deny-file hit
or key outside the --allow-prefix allowlist.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

# Runs both as `python evals/run_golden.py` (sys.path[0] = evals/) and as an
# import from tests; make the repo root importable either way.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ask import Ask, _stored_hashing_dim  # noqa: E402
from vecstore import FastEmbedEmbedder, HashingEmbedder, VecStore  # noqa: E402


def _build_embedder(name: Optional[str], db_path: str):
    if name == "hashing":
        return HashingEmbedder(dim=_stored_hashing_dim(db_path))
    if name == "fastembed":
        return FastEmbedEmbedder()
    return None  # default (lazy fastembed)


def _slug(s: str) -> str:
    """Section-title slug; must mirror vecstore.chunk_markdown's rule."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def _key_slug(key: str) -> str:
    """Section slug embedded in a chunk key '{path}#{section-slug}-{i}'."""
    if "#" not in key:
        return ""
    part = key.split("#", 1)[1]
    return part.rsplit("-", 1)[0] if "-" in part else part


def _load_gold(path: str) -> list[dict]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - import-guard
        raise RuntimeError(
            "run_golden needs pyyaml: `pip install pyyaml`."
        ) from exc
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a YAML list of gold entries")
    return data


def _load_deny(path: str) -> list[str]:
    with open(path, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def _all_docs(store: VecStore) -> list[tuple[str, str]]:
    """(key, text) for every stored doc; empty when no schema exists yet."""
    import sqlite3

    try:
        rows = store.db.execute(
            f"SELECT key, text FROM {store._docs}"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(r["key"], r["text"]) for r in rows]


def _hit_at_k(hits: list[dict], must_cite_paths: list[str]) -> bool:
    """Any top-k hit whose meta['path'] starts with any given prefix.
    Docs with missing meta / missing path never match."""
    for hit in hits:
        meta = hit.get("meta") or {}
        path = str(meta.get("path") or "")
        if path and any(path.startswith(p) for p in must_cite_paths):
            return True
    return False


def _adversarial_violations(hits: list[dict], entry: dict) -> list[str]:
    prefixes = entry.get("forbid_path_prefixes") or []
    slugs = set(entry.get("forbid_section_slugs") or [])
    rx = entry.get("forbid_text_regex")
    out: list[str] = []
    for hit in hits:
        meta = hit.get("meta") or {}
        path = str(meta.get("path") or "")
        key = hit["key"]
        if path and any(path.startswith(p) for p in prefixes):
            out.append(f"{key}: forbidden path prefix")
        elif slugs and (
            _key_slug(key) in slugs or _slug(str(meta.get("section") or "")) in slugs
        ):
            out.append(f"{key}: forbidden section slug")
        elif rx and re.search(rx, hit["text"]):
            out.append(f"{key}: matches forbidden regex {rx!r}")
    return out


def _parse_args(argv: Optional[list[str]]):
    import argparse

    p = argparse.ArgumentParser(
        prog="run_golden",
        description="Golden-set eval harness for a vecstore index.",
    )
    p.add_argument("--db", required=True, help="SQLite file path")
    p.add_argument("--gold", required=True, help="YAML file of gold entries")
    p.add_argument("--k", type=int, default=8, help="retrieval depth for hit@k")
    p.add_argument(
        "--ask",
        action="store_true",
        help="also grade LLM answers (uses $VECSTORE_ASK_CMD)",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="min passing questions for exit 0 (default: all must pass)",
    )
    p.add_argument(
        "--embedder",
        choices=["hashing", "fastembed"],
        default=None,
        help="hashing = deps-free fallback; default = local ONNX model",
    )
    p.add_argument(
        "--adversarial",
        action="store_true",
        help="entries PASS iff nothing forbidden is retrieved at k=20",
    )
    p.add_argument(
        "--deny-file",
        dest="deny_file",
        help="file of forbidden literal terms, one per line (case-insensitive)",
    )
    p.add_argument(
        "--allow-prefix",
        dest="allow_prefix",
        action="append",
        default=[],
        metavar="P",
        help="key path-prefix allowlist; repeatable — any key outside exits 2",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    gold = _load_gold(args.gold)
    store = VecStore(args.db, embedder=_build_embedder(args.embedder, args.db))
    docs = _all_docs(store)

    # Key-prefix audit: every key's path part (before '#') must start with at
    # least one allowed prefix.
    prefix_violations: list[str] = []
    if args.allow_prefix:
        for key, _text in docs:
            path_part = key.split("#", 1)[0]
            if not any(path_part.startswith(p) for p in args.allow_prefix):
                prefix_violations.append(key)

    # Deny scan, part 1: the entire docs table, case-insensitive.
    deny_terms = _load_deny(args.deny_file) if args.deny_file else []
    deny_hits: list[str] = []
    for key, text in docs if deny_terms else []:
        low = text.lower()
        for term in deny_terms:
            if term.lower() in low:
                deny_hits.append(f"term {term!r} in doc {key}")

    asker = Ask(store) if args.ask else None
    passes = 0
    for entry in gold:
        question = str(entry.get("question") or "")
        k = 20 if args.adversarial else args.k
        hits = store.search(question, k=k)

        # Deny scan, part 2: every top-k retrieval output.
        for hit in hits if deny_terms else []:
            low = hit["text"].lower()
            for term in deny_terms:
                if term.lower() in low:
                    deny_hits.append(
                        f"term {term!r} retrieved as {hit['key']} "
                        f"for question {question!r}"
                    )

        detail = ""
        if args.adversarial:
            violations = _adversarial_violations(hits, entry)
            ok = not violations
            detail = "; ".join(violations)
        else:
            must_cite = entry.get("must_cite_paths") or []
            ok = _hit_at_k(hits, must_cite) if must_cite else True
            if not ok:
                detail = f"no top-{k} hit under any of {must_cite}"
            if ok and asker is not None:
                out = asker.ask(question, k=k)
                answer = out["answer"]
                valid = {c["n"] for c in out["citations"]}
                cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
                missing = [
                    s for s in (entry.get("must_contain") or []) if s not in answer
                ]
                if not cited <= valid:
                    ok = False
                    detail = f"answer cites unknown [n]: {sorted(cited - valid)}"
                elif missing:
                    ok = False
                    detail = f"answer missing must_contain: {missing}"

        if ok:
            passes += 1
        line = f"{'PASS' if ok else 'FAIL'}  {question}"
        if detail:
            line += f"  ({detail})"
        print(line)

    total = len(gold)
    print(f"golden: {passes}/{total}")

    deny_hits = list(dict.fromkeys(deny_hits))
    if deny_hits or prefix_violations:
        for v in deny_hits:
            print(f"DENY  {v}")
        for key in prefix_violations:
            print(f"PREFIX  key outside allowlist: {key}")
        return 2

    threshold = args.threshold if args.threshold is not None else total
    return 0 if passes >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
