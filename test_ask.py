"""Tests for the cited-ask layer (T0.4, ask.py). LLM stubbed — no network.
Reuses the seeded store helper from the T0.3 where-filter tests."""

from ask import Ask
from test_vecstore import _seeded


def test_context_numbering_and_citation_mapping():
    s = _seeded()  # 3 chunks from T0.3 helper
    out = Ask(s, llm=lambda p: "answer [2]").ask("ups", k=3)
    assert [c["n"] for c in out["citations"]] == [1, 2, 3]
    assert out["citations"][0]["key"] == s.search("ups", k=1)[0]["key"]
    assert out["answer"] == "answer [2]"


def test_prompt_has_numbered_blocks_and_grounding_instruction():
    seen = {}
    Ask(_seeded(), llm=lambda p: seen.setdefault("p", p) or "x").ask("ups", k=2)
    assert "[1] " in seen["p"] and "[2] " in seen["p"]
    assert "Answer only from the provided context" in seen["p"]


# ---------------------------------------------------------------------------
# Regression — Stage 0 review fixes
# ---------------------------------------------------------------------------

import pytest


def test_default_llm_missing_binary_raises_runtimeerror(monkeypatch):
    """A VECSTORE_ASK_CMD pointing at a nonexistent binary must surface as
    RuntimeError (wrapped shell failure), not a raw FileNotFoundError."""
    monkeypatch.setenv("VECSTORE_ASK_CMD", "/nonexistent-llm-xyz")
    with pytest.raises(RuntimeError):
        Ask(_seeded()).ask("ups", k=1)
