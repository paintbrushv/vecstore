"""Tests for vecstore. Uses HashingEmbedder so no model download / network is
needed — deterministic and CI-safe. Requires `pip install sqlite-vec`."""

import os
import tempfile

import pytest

from vecstore import HashingEmbedder, VecStore, _content_hash, _unit_normalize


def _store(path=":memory:"):
    return VecStore(path, embedder=HashingEmbedder(dim=128))


def test_unit_normalize():
    v = _unit_normalize([3.0, 4.0])
    assert abs((v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-9
    assert _unit_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_content_hash_stable():
    assert _content_hash("hello") == _content_hash("hello")
    assert _content_hash("hello") != _content_hash("world")


def test_add_and_search():
    s = _store()
    s.upsert("a", "the rtx 3090 has 24gb of vram")
    s.upsert("b", "apple m3 max unified memory bandwidth")
    hits = s.search("how much vram on the 3090", k=2)
    assert hits, "expected at least one hit"
    assert hits[0]["key"] == "a"  # token overlap should rank 'a' first
    assert -1.0001 <= hits[0]["score"] <= 1.0001


def test_upsert_skips_unchanged():
    s = _store()
    assert s.upsert("a", "same text") == "added"
    assert s.upsert("a", "same text") == "skipped"
    assert s.upsert("a", "different text") == "updated"


def test_upsert_many_mixed():
    s = _store()
    s.upsert("a", "first")
    results = s.upsert_many(
        [("a", "first", None), ("b", "second", None), ("c", "third", None)]
    )
    assert results[0] == "skipped"
    assert results[1] == "added"
    assert results[2] == "added"
    assert s.stats()["docs"] == 3


def test_delete():
    s = _store()
    s.upsert("a", "doc a")
    s.upsert("b", "doc b")
    assert s.delete("a") is True
    assert s.delete("missing") is False
    assert s.stats()["docs"] == 1
    keys = {h["key"] for h in s.search("doc", k=5)}
    assert "a" not in keys


def test_meta_roundtrip():
    s = _store()
    s.upsert("a", "doc a", meta={"source": "test", "n": 1})
    hit = s.search("doc a", k=1)[0]
    assert hit["meta"] == {"source": "test", "n": 1}


def test_search_empty_store():
    s = _store()
    assert s.search("anything") == []


def test_embedder_mismatch_guards():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        s1 = VecStore(path, embedder=HashingEmbedder(dim=64))
        s1.upsert("a", "hello")
        s1.close()
        # Reopen with a different embedder dim -> must refuse, not corrupt.
        s2 = VecStore(path, embedder=HashingEmbedder(dim=128))
        with pytest.raises(RuntimeError):
            s2.upsert("b", "world")
        s2.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_rebuild_same_model_is_stable():
    s = _store()
    s.upsert("a", "rtx 3090 vram")
    s.upsert("b", "apple unified memory")
    before = s.search("vram", k=2)
    n = s.rebuild()
    assert n == 2
    after = s.search("vram", k=2)
    assert [h["key"] for h in before] == [h["key"] for h in after]


def test_persistence_and_incremental_reopen():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        s1 = VecStore(path, embedder=HashingEmbedder(dim=128))
        s1.upsert("a", "persisted doc about gpus")
        s1.close()
        # Reopen: same content should skip (incremental sync is cheap).
        s2 = VecStore(path, embedder=HashingEmbedder(dim=128))
        assert s2.upsert("a", "persisted doc about gpus") == "skipped"
        assert s2.search("gpus", k=1)[0]["key"] == "a"
        s2.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)
