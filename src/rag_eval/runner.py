"""Run a golden dataset through a RAG system and score the results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import metrics
from .systems import RagSystem, SystemResult


@dataclass
class GoldenItem:
    """One question in the golden dataset."""

    id: str
    question: str
    expected_answer: str
    expected_species: str | None
    relevant_chunk_ids: list[int]
    should_abstain: bool
    category: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenItem":
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            expected_answer=str(data.get("expected_answer", "")),
            expected_species=data.get("expected_species"),
            relevant_chunk_ids=[int(i) for i in data.get("relevant_chunk_ids", [])],
            should_abstain=bool(data.get("should_abstain", False)),
            category=str(data.get("category", "factoid")),
        )


@dataclass
class ItemResult:
    """Per-question outcome: what the system did and how it scored."""

    id: str
    question: str
    category: str
    expected_answer: str
    expected_species: str | None
    should_abstain: bool
    predicted_answer: str
    predicted_species: str | None
    abstained: bool
    retrieved_chunk_ids: list[int]
    relevant_chunk_ids: list[int]
    recall_at_k: float | None
    hit_rate: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    token_f1: float
    exact_match: float
    rouge_l: float
    citation_correct: bool
    abstention_correct: bool


def load_dataset(path: str | Path) -> list[GoldenItem]:
    """Load a golden dataset JSON file into a list of GoldenItem."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Dataset {path} must be a JSON list of items.")
    return [GoldenItem.from_dict(item) for item in data]


def score_item(item: GoldenItem, result: SystemResult, k: int) -> ItemResult:
    """Score one system answer against one golden item."""
    # Normalise em dashes in the stored answer so the JSON and Markdown
    # reports match (document-qa-rag's abstain text contains one). This
    # does not change the metrics: the answer normaliser treats a dash
    # as a separator either way.
    answer = result.answer.replace("\u2014", "-")
    if item.should_abstain:
        token_f1 = 1.0 if result.abstained else 0.0
        exact = 1.0 if result.abstained else 0.0
        rouge = 1.0 if result.abstained else 0.0
    elif result.abstained:
        # Abstaining on an answerable question scores zero on answers.
        token_f1 = 0.0
        exact = 0.0
        rouge = 0.0
    else:
        token_f1 = metrics.token_f1(answer, item.expected_answer)
        exact = metrics.exact_match(answer, item.expected_answer)
        rouge = metrics.rouge_l(answer, item.expected_answer)
    if item.relevant_chunk_ids:
        recall = metrics.recall_at_k(result.retrieved_chunk_ids, item.relevant_chunk_ids, k)
        hit = metrics.hit_rate(result.retrieved_chunk_ids, item.relevant_chunk_ids, k)
        rr = metrics.reciprocal_rank(result.retrieved_chunk_ids, item.relevant_chunk_ids)
        ndcg = metrics.ndcg_at_k(result.retrieved_chunk_ids, item.relevant_chunk_ids, k)
    else:
        # No relevant chunks exist (out-of-scope items): retrieval is
        # undefined, not zero. Stored as None so it serialises as null
        # and is excluded from aggregation.
        recall = hit = rr = ndcg = None
    return ItemResult(
        id=item.id,
        question=item.question,
        category=item.category,
        expected_answer=item.expected_answer,
        expected_species=item.expected_species,
        should_abstain=item.should_abstain,
        predicted_answer=answer,
        predicted_species=result.cited_species,
        abstained=result.abstained,
        retrieved_chunk_ids=list(result.retrieved_chunk_ids),
        relevant_chunk_ids=list(item.relevant_chunk_ids),
        recall_at_k=recall,
        hit_rate=hit,
        reciprocal_rank=rr,
        ndcg_at_k=ndcg,
        token_f1=token_f1,
        exact_match=exact,
        rouge_l=rouge,
        citation_correct=(
            metrics.citation_correct(result.cited_species, item.expected_species)
            if not item.should_abstain
            else result.cited_species is None
        ),
        abstention_correct=metrics.abstention_correct(result.abstained, item.should_abstain),
    )


def _mean_or_none(values: list[float]) -> float | None:
    """Mean of a list, or None when the list is empty (metric not defined)."""
    return metrics.mean(values) if values else None


def _aggregate(items: list[ItemResult], k: int) -> dict[str, Any]:
    """Aggregate metrics over a list of scored items.

    Metrics that have no applicable items (for example retrieval metrics
    for the out-of-scope category, which has no relevant chunks) are
    None, rendered as "n/a", rather than a misleading 0.0.
    """
    retrievable = [r for r in items if r.relevant_chunk_ids]
    answerable = [r for r in items if not r.should_abstain]
    cited = [r for r in answerable if r.predicted_species is not None]
    return {
        "n": len(items),
        f"recall@{k}": _mean_or_none([r.recall_at_k for r in retrievable]),
        "mrr": _mean_or_none([r.reciprocal_rank for r in retrievable]),
        f"ndcg@{k}": _mean_or_none([r.ndcg_at_k for r in retrievable]),
        f"hit_rate@{k}": _mean_or_none([r.hit_rate for r in retrievable]),
        "answer_token_f1": _mean_or_none([r.token_f1 for r in answerable]),
        "answer_exact_match": _mean_or_none([r.exact_match for r in answerable]),
        "answer_rouge_l": _mean_or_none([r.rouge_l for r in answerable]),
        "citation_accuracy": _mean_or_none(
            [1.0 if r.citation_correct else 0.0 for r in cited]
        ),
        "citation_coverage": len(cited) / len(answerable) if answerable else None,
        "abstention_accuracy": metrics.mean([1.0 if r.abstention_correct else 0.0 for r in items]),
    }


def run_evaluation(
    system: RagSystem,
    dataset: list[GoldenItem],
    k: int = 5,
) -> dict[str, Any]:
    """Run every dataset item through ``system`` and aggregate the scores.

    Returns a report dict with ``system``, ``k``, ``metrics`` (overall),
    ``by_category`` (one aggregate per category) and ``items`` (per-item
    results). Category splits expose gaps such as paraphrase vs factoid.
    """
    results = [score_item(item, system.ask(item.question, k=k), k) for item in dataset]
    categories: dict[str, list[ItemResult]] = {}
    for result in results:
        categories.setdefault(result.category, []).append(result)
    return {
        "system": system.name,
        "k": k,
        "n_items": len(results),
        "metrics": _aggregate(results, k),
        "by_category": {name: _aggregate(rows, k) for name, rows in categories.items()},
        "items": [asdict(r) for r in results],
    }


def _fmt(value: float | None) -> str:
    """Format a metric value, or "n/a" when it is not defined."""
    return f"{value:.3f}" if value is not None else "n/a"


def summary_table(report: dict[str, Any]) -> str:
    """Render the overall metrics as a plain-text summary table."""
    k = report["k"]
    m = report["metrics"]
    rows = [
        ("System", report["system"]),
        ("Questions", str(report["n_items"])),
        (f"Recall@{k}", _fmt(m[f"recall@{k}"])),
        ("MRR", _fmt(m["mrr"])),
        (f"nDCG@{k}", _fmt(m[f"ndcg@{k}"])),
        (f"Hit rate@{k}", _fmt(m[f"hit_rate@{k}"])),
        ("Answer token F1", _fmt(m["answer_token_f1"])),
        ("Answer exact match", _fmt(m["answer_exact_match"])),
        ("Answer ROUGE-L", _fmt(m["answer_rouge_l"])),
        ("Citation accuracy", _fmt(m["citation_accuracy"])),
        ("Citation coverage", _fmt(m["citation_coverage"])),
        ("Abstention accuracy", _fmt(m["abstention_accuracy"])),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"{label:<{width}}  {value}" for label, value in rows]
    return "\n".join(lines)


def markdown_report(report: dict[str, Any]) -> str:
    """Render the report as Markdown (overall table plus category split)."""
    k = report["k"]
    m = report["metrics"]
    lines = [
        "# RAG evaluation report",
        "",
        f"System: `{report['system']}`, k={k}, questions: {report['n_items']}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Recall@{k} | {_fmt(m[f'recall@{k}'])} |",
        f"| MRR | {_fmt(m['mrr'])} |",
        f"| nDCG@{k} | {_fmt(m[f'ndcg@{k}'])} |",
        f"| Hit rate@{k} | {_fmt(m[f'hit_rate@{k}'])} |",
        f"| Answer token F1 | {_fmt(m['answer_token_f1'])} |",
        f"| Answer exact match | {_fmt(m['answer_exact_match'])} |",
        f"| Answer ROUGE-L | {_fmt(m['answer_rouge_l'])} |",
        f"| Citation accuracy | {_fmt(m['citation_accuracy'])} |",
        f"| Citation coverage | {_fmt(m['citation_coverage'])} |",
        f"| Abstention accuracy | {_fmt(m['abstention_accuracy'])} |",
        "",
        "## By category",
        "",
        "| Category | n | " + f"Recall@{k} | MRR | Answer F1 | Abstention |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, cm in report["by_category"].items():
        lines.append(
            f"| {name} | {cm['n']} | {_fmt(cm[f'recall@{k}'])} | {_fmt(cm['mrr'])} | "
            f"{_fmt(cm['answer_token_f1'])} | {_fmt(cm['abstention_accuracy'])} |"
        )
    lines += ["", "## Per-question results", ""]
    for item in report["items"]:
        # [OK] means the answer, citation and abstention are all right:
        # for answerable items the answer must actually score (token F1
        # above zero), not just have the right citation. For out-of-scope
        # items there is no answer to score, so abstention + citation
        # (no citation given) is the whole check.
        if item["should_abstain"]:
            ok = item["abstention_correct"] and item["citation_correct"]
        else:
            ok = (
                item["abstention_correct"]
                and item["citation_correct"]
                and item["token_f1"] > 0
            )
        mark = "OK" if ok else "CHECK"
        # Answers are already normalised in score_item; this keeps reports
        # built from older JSON em-dash free as well.
        answer = item["predicted_answer"].replace("\u2014", "-")
        lines.append(
            f"- [{mark}] {item['id']} ({item['category']}): {item['question']} "
            f"-> {answer}"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write the JSON and Markdown reports into ``out_dir`` and return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "GoldenItem",
    "ItemResult",
    "load_dataset",
    "markdown_report",
    "run_evaluation",
    "score_item",
    "summary_table",
    "write_reports",
]
