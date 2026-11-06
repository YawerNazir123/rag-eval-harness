"""rag-eval-harness: a small, reusable evaluation harness for RAG systems."""

from .metrics import (
    citation_correct,
    abstention_correct,
    exact_match,
    hit_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)
from .runner import GoldenItem, load_dataset, run_evaluation
from .systems import Citation, DocQARagSystem, HttpRagSystem, RagSystem, StubSystem, SystemResult

__all__ = [
    "Citation",
    "DocQARagSystem",
    "GoldenItem",
    "HttpRagSystem",
    "RagSystem",
    "StubSystem",
    "SystemResult",
    "abstention_correct",
    "citation_correct",
    "exact_match",
    "hit_rate",
    "load_dataset",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "rouge_l",
    "run_evaluation",
    "token_f1",
]

__version__ = "0.1.0"
