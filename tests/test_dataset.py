"""Validity tests for the golden dataset.

Every non-abstain item must be grounded: its expected answer substring
must appear verbatim in the cited chunk text of the real document-qa-rag
index, and every relevant chunk id must exist. Out-of-scope items must
have no expected species and must ask about a species that is genuinely
absent from the corpus.
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden.json"
CHUNKS_PATH = (
    Path.home()
    / "workspace"
    / "goals"
    / "get-hired-as-an-ai-ml-engineer"
    / "hidden_files"
    / "document-qa-rag"
    / "index"
    / "chunks.json"
)
# Committed fixture: the same 744-slot index, but only the 15 chunks the
# golden set cites carry real text; the rest are empty placeholders.
# It exists so CI verifies dataset grounding even though the real index
# is not checked out there. When the real index is present it wins, so
# local runs still test against the full corpus. If the golden set cites
# new chunks, regenerate this file from the real index.
FIXTURE_CHUNKS_PATH = PROJECT_ROOT / "tests" / "fixtures" / "chunks.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunks():
    if CHUNKS_PATH.exists():
        return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if FIXTURE_CHUNKS_PATH.exists():
        return json.loads(FIXTURE_CHUNKS_PATH.read_text(encoding="utf-8"))
    pytest.skip("document-qa-rag index not available (no real index, no fixture)")


def test_dataset_size_and_ids(golden):
    assert 30 <= len(golden) <= 50
    ids = [item["id"] for item in golden]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    categories = {item["category"] for item in golden}
    assert {"factoid", "paraphrase", "out_of_scope"} <= categories


def test_required_fields(golden):
    required = {"id", "question", "expected_answer", "expected_species",
                "relevant_chunk_ids", "should_abstain", "category"}
    for item in golden:
        assert required <= set(item), f"missing fields in {item.get('id')}"
        assert item["question"].strip()
        assert isinstance(item["should_abstain"], bool)


def test_abstain_items_have_no_species(golden):
    abstain = [i for i in golden if i["should_abstain"]]
    assert 4 <= len(abstain) <= 6
    for item in abstain:
        assert item["expected_species"] is None
        assert item["relevant_chunk_ids"] == []
        assert item["category"] == "out_of_scope"


def test_paraphrase_items_present(golden):
    assert 5 <= len([i for i in golden if i["category"] == "paraphrase"]) <= 8


def test_relevant_chunk_ids_exist(golden, chunks):
    # Chunk ids are positions in the index list (see README: the stored
    # metadata chunk_id is 0 for every chunk, so it cannot identify one).
    for item in golden:
        for chunk_id in item["relevant_chunk_ids"]:
            assert 0 <= chunk_id < len(chunks), f"{item['id']}: chunk {chunk_id} missing"


def test_expected_answer_in_cited_chunk_text(golden, chunks):
    for item in golden:
        if item["should_abstain"]:
            continue
        assert item["expected_answer"], f"{item['id']}: empty expected answer"
        assert item["relevant_chunk_ids"], f"{item['id']}: no relevant chunks"
        found = any(
            item["expected_answer"] in chunks[cid]["text"]
            for cid in item["relevant_chunk_ids"]
        )
        assert found, f"{item['id']}: expected answer not in cited chunk text"


def test_expected_species_matches_cited_chunk(golden, chunks):
    for item in golden:
        if item["should_abstain"]:
            continue
        species = {chunks[cid]["metadata"]["common_name"] for cid in item["relevant_chunk_ids"]}
        assert item["expected_species"] in species, (
            f"{item['id']}: expected species {item['expected_species']} "
            f"not among cited chunks {species}"
        )


def test_out_of_scope_species_absent_from_corpus(golden, chunks):
    # Empty common names are fixture placeholders, not species.
    corpus_names = {c["metadata"]["common_name"].lower() for c in chunks if c["metadata"].get("common_name")}
    # Species named in out-of-scope questions must not appear in the corpus.
    for item in golden:
        if not item["should_abstain"]:
            continue
        question = item["question"].lower()
        for name in corpus_names:
            assert name not in question, (
                f"{item['id']}: corpus species {name!r} appears in an "
                "out-of-scope question"
            )
