"""Tests for the golden eval harness (T0.4, evals/run_golden.py).

Builds a tmp HashingEmbedder db with one chunk and exercises the exit-code
contract: 0 = pass, 1 = golden threshold fail, 2 = deny-file hit or key
outside the --allow-prefix set. Zero network.
"""

import importlib.util
from pathlib import Path

import pytest

from vecstore import HashingEmbedder, VecStore

_RUN_GOLDEN_PATH = Path(__file__).resolve().parent / "evals" / "run_golden.py"
_spec = importlib.util.spec_from_file_location("run_golden", _RUN_GOLDEN_PATH)
run_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_golden)


CHUNK_KEY = "logs/digests/2026-07-05.md#deal-alerts-0"
CHUNK_TEXT = "STRONG BUY: Square D breaker lot, 62% below fair value."
CHUNK_META = {
    "path": "logs/digests/2026-07-05.md",
    "section": "Deal Alerts",
    "date": "2026-07-05",
    "source": "digests",
}

GOLD_HIT = """\
- question: "strong buy breaker"
  must_cite_paths: ["logs/digests/"]
"""

GOLD_MISS = """\
- question: "strong buy breaker"
  must_cite_paths: ["docs/never/"]
"""


@pytest.fixture
def db(tmp_path):
    """Tmp db built with the same embedder `--embedder hashing` constructs
    (HashingEmbedder() default dim), so the model-id guard matches."""
    path = str(tmp_path / "golden.db")
    store = VecStore(path, embedder=HashingEmbedder())
    store.upsert(CHUNK_KEY, CHUNK_TEXT, meta=CHUNK_META)
    return path


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _argv(db, gold, *extra):
    return ["--db", db, "--gold", gold, "--k", "8",
            "--threshold", "1", "--embedder", "hashing", *extra]


def test_golden_hit_returns_0(db, tmp_path):
    gold = _write(tmp_path, "gold.yaml", GOLD_HIT)
    assert run_golden.main(_argv(db, gold)) == 0


def test_golden_miss_returns_1(db, tmp_path):
    gold = _write(tmp_path, "gold.yaml", GOLD_MISS)
    assert run_golden.main(_argv(db, gold)) == 1


def test_deny_file_hit_returns_2(db, tmp_path):
    gold = _write(tmp_path, "gold.yaml", GOLD_HIT)
    deny = _write(tmp_path, "deny.txt", "square d\n")
    assert run_golden.main(_argv(db, gold, "--deny-file", deny)) == 2


def test_allow_prefix_audit(db, tmp_path):
    gold = _write(tmp_path, "gold.yaml", GOLD_HIT)
    assert run_golden.main(_argv(db, gold, "--allow-prefix", "logs/digests/")) == 0
    assert run_golden.main(_argv(db, gold, "--allow-prefix", "docs/never/")) == 2
