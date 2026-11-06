"""Tests for the runner, the CLI, and the HTTP/stub adapters."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rag_eval.runner import GoldenItem, load_dataset, markdown_report, run_evaluation, score_item
from rag_eval.systems import Citation, HttpRagSystem, StubSystem, SystemResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _item(**overrides):
    data = {
        "id": "q01",
        "question": "What is the Kashmiri name of the Barn Owl?",
        "expected_answer": "Kashmiri name: Rata Mogul",
        "expected_species": "Barn Owl",
        "relevant_chunk_ids": [10],
        "should_abstain": False,
        "category": "factoid",
    }
    data.update(overrides)
    return GoldenItem.from_dict(data)


def test_score_item_perfect_answer():
    result = SystemResult(
        answer="Barn Owl: Kashmiri name: Rata Mogul.",
        citations=[Citation(species="Barn Owl")],
        abstained=False,
        retrieved_chunk_ids=[3, 10, 7],
    )
    scored = score_item(_item(), result, k=5)
    assert scored.recall_at_k == pytest.approx(1.0)
    assert scored.hit_rate == pytest.approx(1.0)
    assert scored.reciprocal_rank == pytest.approx(0.5)  # relevant at rank 2
    assert scored.citation_correct is True
    assert scored.abstention_correct is True
    assert scored.token_f1 > 0.5


def test_score_item_wrong_species_and_abstention():
    result = SystemResult(
        answer="I don't know.",
        citations=[],
        abstained=True,
        retrieved_chunk_ids=[3, 4],
    )
    scored = score_item(_item(), result, k=5)
    assert scored.recall_at_k == pytest.approx(0.0)
    assert scored.token_f1 == pytest.approx(0.0)
    assert scored.citation_correct is False
    assert scored.abstention_correct is False


def test_score_item_out_of_scope_abstention():
    item = _item(
        id="q99",
        question="What is the Kashmiri name of the African Penguin?",
        expected_answer="",
        expected_species=None,
        relevant_chunk_ids=[],
        should_abstain=True,
        category="out_of_scope",
    )
    result = SystemResult(answer="I don't know.", abstained=True, retrieved_chunk_ids=[])
    scored = score_item(item, result, k=5)
    assert scored.abstention_correct is True
    assert scored.citation_correct is True


def test_run_evaluation_category_split():
    system = StubSystem(
        answers={
            "What is the Kashmiri name of the Barn Owl?": SystemResult(
                answer="Kashmiri name: Rata Mogul",
                citations=[Citation(species="Barn Owl")],
                retrieved_chunk_ids=[10],
            )
        }
    )
    dataset = [_item(), _item(id="q02", question="Other question?", category="paraphrase")]
    report = run_evaluation(system, dataset, k=5)
    assert report["n_items"] == 2
    assert set(report["by_category"]) == {"factoid", "paraphrase"}
    assert report["by_category"]["factoid"]["recall@5"] == pytest.approx(1.0)
    assert report["by_category"]["paraphrase"]["recall@5"] == pytest.approx(0.0)


def test_load_dataset_rejects_non_list(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(bad)


def test_cli_stub_end_to_end(tmp_path):
    dataset = [
        {
            "id": "t01",
            "question": "What is the Kashmiri name of the Barn Owl?",
            "expected_answer": "Kashmiri name: Rata Mogul",
            "expected_species": "Barn Owl",
            "relevant_chunk_ids": [10],
            "should_abstain": False,
            "category": "factoid",
        },
        {
            "id": "t02",
            "question": "What is the Kashmiri name of the African Penguin?",
            "expected_answer": "",
            "expected_species": None,
            "relevant_chunk_ids": [],
            "should_abstain": True,
            "category": "out_of_scope",
        },
        {
            "id": "t03",
            "question": "What is the length of the Barn Owl?",
            "expected_answer": "Length: 33-39 cm",
            "expected_species": "Barn Owl",
            "relevant_chunk_ids": [10],
            "should_abstain": False,
            "category": "factoid",
        },
    ]
    dataset_path = tmp_path / "tiny.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    out_dir = tmp_path / "results"
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src"), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, "-m", "rag_eval", "run",
         "--dataset", str(dataset_path), "--system", "stub",
         "--out", str(out_dir), "--k", "5"],
        capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "Recall@5" in proc.stdout
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["n_items"] == 3
    assert report["system"] == "stub"
    assert (out_dir / "report.md").exists()
    assert "By category" in (out_dir / "report.md").read_text(encoding="utf-8")


def test_cli_http_requires_endpoint(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "rag_eval", "run",
         "--dataset", str(PROJECT_ROOT / "data" / "golden.json"),
         "--system", "http"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(PROJECT_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 2


class _AskHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = {
            "answer": f"Echo: {body.get('question', '')}",
            "citations": [{"species": "Barn Owl", "source": "test", "page": 1}],
            "abstained": False,
            "retrieved_chunk_ids": [10, 3],
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_http_system_tolerant_parsing():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AskHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        system = HttpRagSystem(f"http://127.0.0.1:{server.server_address[1]}/ask")
        result = system.ask("What is the Kashmiri name of the Barn Owl?", k=5)
        assert result.answer.startswith("Echo:")
        assert result.cited_species == "Barn Owl"
        assert result.retrieved_chunk_ids == [10, 3]
        assert result.abstained is False
    finally:
        server.shutdown()
        server.server_close()

def test_score_item_failed_abstention_scores_zero_answer_metrics():
    """An out-of-scope item the system answered (not abstained) scores 0.0
    on token F1, exact match and ROUGE-L, like the other answer metrics."""
    item = _item(
        id="q33",
        question="Which family does the Ostrich belong to?",
        expected_answer="",
        expected_species=None,
        relevant_chunk_ids=[],
        should_abstain=True,
        category="out_of_scope",
    )
    result = SystemResult(
        answer="Hoopoe: Category: Other.",
        citations=[Citation(species="Hoopoe")],
        abstained=False,
        retrieved_chunk_ids=[5],
    )
    scored = score_item(item, result, k=5)
    assert scored.token_f1 == pytest.approx(0.0)
    assert scored.exact_match == pytest.approx(0.0)
    assert scored.rouge_l == pytest.approx(0.0)
    assert scored.abstention_correct is False


def test_score_item_correct_abstention_scores_full_answer_metrics():
    item = _item(
        id="q31",
        question="What is the Kashmiri name of the African Penguin?",
        expected_answer="",
        expected_species=None,
        relevant_chunk_ids=[],
        should_abstain=True,
        category="out_of_scope",
    )
    result = SystemResult(
        answer="I don't know - not in the documents",
        citations=[],
        abstained=True,
        retrieved_chunk_ids=[],
    )
    scored = score_item(item, result, k=5)
    assert scored.token_f1 == pytest.approx(1.0)
    assert scored.exact_match == pytest.approx(1.0)
    assert scored.rouge_l == pytest.approx(1.0)


def test_score_item_out_of_scope_retrieval_metrics_are_none():
    """Per-item retrieval metrics are None (not 0.0) when an item has no
    relevant chunks, and the None values survive a JSON round-trip."""
    item = _item(
        id="q99",
        question="What is the Kashmiri name of the African Penguin?",
        expected_answer="",
        expected_species=None,
        relevant_chunk_ids=[],
        should_abstain=True,
        category="out_of_scope",
    )
    result = SystemResult(answer="I don't know.", abstained=True, retrieved_chunk_ids=[])
    scored = score_item(item, result, k=5)
    assert scored.recall_at_k is None
    assert scored.hit_rate is None
    assert scored.reciprocal_rank is None
    assert scored.ndcg_at_k is None
    round_tripped = json.loads(json.dumps({"items": [scored.__dict__]}))
    assert round_tripped["items"][0]["recall_at_k"] is None


def test_score_item_normalises_em_dash_in_stored_answer():
    """The stored predicted answer uses a hyphen, so the JSON report and
    the Markdown report agree (no raw em dash in either)."""
    item = _item()
    result = SystemResult(
        answer="I don't know \u2014 not in the documents",
        citations=[],
        abstained=True,
        retrieved_chunk_ids=[],
    )
    scored = score_item(item, result, k=5)
    assert "\u2014" not in scored.predicted_answer
    assert scored.predicted_answer == "I don't know - not in the documents"


def test_markdown_flag_needs_a_scoring_answer():
    """[OK] requires a non-zero token F1 on answerable items: right
    citation plus right abstention is not enough when the answer itself
    scores 0.0 (the q27 case)."""
    item = _item(
        id="q27",
        question="How big is the Alexandrine Parakeet from beak to tail?",
        expected_answer="Length: 56\u201362 cm",
        expected_species="Alexandrine Parakeet",
        relevant_chunk_ids=[0],
        should_abstain=False,
        category="paraphrase",
    )
    result = SystemResult(
        answer="Alexandrine Parakeet: Family: Psittaculidae.",
        citations=[Citation(species="Alexandrine Parakeet")],
        abstained=False,
        retrieved_chunk_ids=[0],
    )
    scored = score_item(item, result, k=5)
    assert scored.token_f1 == pytest.approx(0.0)
    assert scored.citation_correct is True
    assert scored.abstention_correct is True
    system = StubSystem(answers={item.question: result})
    report = run_evaluation(system, [item], k=5)
    md = markdown_report(report)
    assert "- [CHECK] q27 " in md


class _ErrorHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(500)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "11")
        self.end_headers()
        self.wfile.write(b"boom error!")

    def log_message(self, *args):
        pass


def _run_server(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_http_system_raises_descriptive_error_on_non_200():
    server = _run_server(_ErrorHandler)
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/ask"
        system = HttpRagSystem(endpoint)
        with pytest.raises(RuntimeError) as exc_info:
            system.ask("What is the Kashmiri name of the Barn Owl?", k=5)
        message = str(exc_info.value)
        assert endpoint in message
        assert "500" in message
    finally:
        server.shutdown()
        server.server_close()


class _BadJsonHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = b"this is not json"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_http_system_raises_descriptive_error_on_bad_json():
    server = _run_server(_BadJsonHandler)
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/ask"
        system = HttpRagSystem(endpoint)
        with pytest.raises(RuntimeError) as exc_info:
            system.ask("What is the Kashmiri name of the Barn Owl?", k=5)
        assert endpoint in str(exc_info.value)
    finally:
        server.shutdown()
        server.server_close()


def test_out_of_scope_category_has_no_retrieval_metrics():
    """Retrieval metrics are None (not 0.0) when a category has no relevant chunks."""
    item = _item(
        id="q99",
        question="What is the Kashmiri name of the African Penguin?",
        expected_answer="",
        expected_species=None,
        relevant_chunk_ids=[],
        should_abstain=True,
        category="out_of_scope",
    )
    report = run_evaluation(StubSystem(), [item], k=5)
    assert report["by_category"]["out_of_scope"]["recall@5"] is None
    assert report["by_category"]["out_of_scope"]["answer_token_f1"] is None
    assert report["by_category"]["out_of_scope"]["abstention_accuracy"] == pytest.approx(1.0)
    # Per-item retrieval metrics are null too, not 0.0.
    assert report["items"][0]["recall_at_k"] is None
    assert report["items"][0]["hit_rate"] is None
    assert report["items"][0]["reciprocal_rank"] is None
    assert report["items"][0]["ndcg_at_k"] is None
