"""Tests for the markdown-section chunker (T0.1) and chunked ingest with
stale-chunk cleanup (T0.2). Uses HashingEmbedder so no model download /
network is needed — deterministic and CI-safe."""

from vecstore import HashingEmbedder, VecStore, chunk_markdown

DIGEST = """====================================================================
SURPLUS EQUIPMENT MARKET DIGEST — 2026-07-05
====================================================================

── Deal Alerts ────────────────────────────────────────────────────
- STRONG BUY: Square D breaker lot, 62% below fair value.

── Electrical ─────────────────────────────────────────────────────
Panels tightened week over week.

── UPS & Power ────────────────────────────────────────────────────
Eaton 9PX lots cleared at 0.4x landed.
"""


# ---------------------------------------------------------------------------
# T0.1 — chunk_markdown
# ---------------------------------------------------------------------------


def test_banner_split_one_chunk_per_section():
    chunks = chunk_markdown(DIGEST, "logs/digests/2026-07-05.md")
    assert [c.meta["section"] for c in chunks] == ["Deal Alerts", "Electrical", "UPS & Power"]
    assert chunks[2].key == "logs/digests/2026-07-05.md#ups-power-0"
    assert all(c.meta["date"] == "2026-07-05" for c in chunks)
    assert all(c.meta["source"] == "digests" for c in chunks)


def test_headingless_banner_doc_yields_named_sections():
    assert not any(l.lstrip().startswith("#") for l in DIGEST.splitlines())  # zero markdown headings
    chunks = chunk_markdown(DIGEST, "logs/digests/2026-07-05.md")
    assert chunks and all(c.meta["section"] != "preamble" for c in chunks)


def test_oversize_section_splits_with_overlap():
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(20))
    chunks = chunk_markdown("# Big\n\n## Long Section\n\n" + body,
                            "logs/digests/2026-06-21.md", max_tokens=400, overlap=50)
    long = [c for c in chunks if "#long-section-" in c.key]
    assert len(long) >= 3
    assert all(len(c.text) // 4 <= 400 for c in long)
    assert long[0].text.split("\n\n")[-1] in long[1].text  # overlap carried


def test_keys_stable_across_reruns():
    a = chunk_markdown(DIGEST, "logs/digests/2026-07-05.md")
    b = chunk_markdown(DIGEST, "logs/digests/2026-07-05.md")
    assert [c.key for c in a] == [c.key for c in b]


def test_44kb_sectioned_digest_every_section_represented():
    text = "# Digest 2026-07-01\n\n" + "".join(
        f"## Section {i:02d}\n\n" + f"Deal line {i} " * 400 + "\n\n" for i in range(18))
    assert len(text) > 40_000
    chunks = chunk_markdown(text, "logs/digests/2026-07-01.md")
    assert {c.meta["section"] for c in chunks} == {f"Section {i:02d}" for i in range(18)}


# ---------------------------------------------------------------------------
# T0.2 — VecStore.sync_chunks
# ---------------------------------------------------------------------------


def test_sync_chunks_removes_orphans():
    s = VecStore(":memory:", embedder=HashingEmbedder(dim=128))
    v1 = chunk_markdown("# T\n\n## A\n\naaa\n\n## B\n\nbbb\n", "d/2026-07-01.md")
    s.sync_chunks("d/2026-07-01.md", v1)
    v2 = chunk_markdown("# T\n\n## A\n\naaa\n", "d/2026-07-01.md")  # section B gone
    res = s.sync_chunks("d/2026-07-01.md", v2)
    assert res == {"added": 0, "updated": 0, "skipped": 1, "deleted": 1}
    assert not any("#b-" in h["key"] for h in s.search("bbb", k=10))


def test_sync_chunks_unchanged_doc_all_skipped():
    s = VecStore(":memory:", embedder=HashingEmbedder(dim=128))
    v = chunk_markdown(DIGEST, "logs/digests/2026-07-05.md")
    s.sync_chunks("logs/digests/2026-07-05.md", v)
    res = s.sync_chunks("logs/digests/2026-07-05.md", v)
    assert res["skipped"] == len(v) and res["added"] == res["deleted"] == 0


# ---------------------------------------------------------------------------
# Regression — Stage 0 review fixes
# ---------------------------------------------------------------------------


def test_sync_chunks_refreshes_meta_on_unchanged_text():
    """Re-syncing identical text whose chunks carry new meta must not re-embed
    (counts stay 'skipped'), yet the stored meta is refreshed so where-filters
    see the new values."""
    s = VecStore(":memory:", embedder=HashingEmbedder(dim=128))
    path = "logs/digests/2026-07-05.md"
    s.sync_chunks(path, chunk_markdown(DIGEST, path))  # meta source == "digests"

    renamed = chunk_markdown(DIGEST, path)  # same text, same keys
    for c in renamed:
        c.meta["source"] = "newname"
    res = s.sync_chunks(path, renamed)
    assert res["skipped"] == len(renamed)
    assert res["added"] == res["updated"] == 0

    hits = s.search("Eaton 9PX lots cleared", k=10, where={"source": "newname"})
    assert any(h["key"] == f"{path}#ups-power-0" for h in hits)
    assert s.search("Eaton 9PX lots cleared", k=10, where={"source": "digests"}) == []


def test_sync_chunks_migrates_whole_file_doc_to_chunks():
    """A legacy whole-file doc stored under the bare path is deleted when the
    same doc is re-ingested chunked — no stale whole-file hit lingers."""
    text = "# T\n\n## A\n\nalpha equipment notes\n\n## B\n\nbeta market notes\n"
    s = VecStore(":memory:", embedder=HashingEmbedder(dim=128))
    s.upsert("d/f.md", text)
    res = s.sync_chunks("d/f.md", chunk_markdown(text, "d/f.md"))
    assert res["deleted"] >= 1
    assert all(h["key"] != "d/f.md" for h in s.search(text, k=10))
